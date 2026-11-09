"""pt (safetensors) -> onnx 导出

yolo26：默认导出 E2E 融合图，单输出 output0 (B, 300, 6)，解码与两阶段 top-k 全部在图内；
--raw 导出原始头输出 (B, 4+nc, 8400)，配合 NMS 路径在外部解码（utils/postprocess.py）。
yolo26-seg：双输出——output0 在检测输出末维追加 nm 个 mask 系数（E2E: (B, 300, 6+nm)；
raw: (B, 4+nc+nm, 8400)），output1 是原型 (B, nm, H/4, W/4)；掩码在外部由
utils/postprocess.process_mask 解码（与官方 val 逐像素一致）。
yolov3-tiny：单输出解码 (B, NA, 5+nc)——obj×cls 与按类 NMS 在外部（utils/postprocess.v3_detections），
--raw 不适用。
yoloe-26 text：默认导出 fuse 后的 E2E 类锁定图；--nms 改出 o2m 密集输出
(B, 4+V+nm, 8400)（像素 xyxy + sigmoid 分数），NMS 在外部（utils/postprocess.nms_per_image）。
--visual-prompt（仅 text 权重）：refer 图 + bbox/mask JSON -> SAVPE vpe -> 同一 fuse 路径；
可与 --classes 组合成文本+视觉混合 prompt（文本在前、视觉在后，官方 get_cls_pe 顺序）。
默认 batch 固定为 1（边缘工具链偏好固定 shape），--dynamic 打开动态 batch。

用法:
    python scripts/export.py --weights weights/yolo26n.safetensors --out weights/yolo26n.onnx
    python scripts/export.py --weights weights/yolo26n.safetensors --out weights/yolo26n_raw.onnx --raw
    python scripts/export.py --weights weights/yolov3-tiny.safetensors --model yolov3-tiny
    python scripts/export.py --weights weights/yoloe-26n-seg.safetensors --classes "person, car"
    python scripts/export.py --weights weights/yoloe-26n-seg.safetensors --classes "person, car" --nms
    python scripts/export.py --weights weights/yoloe-26n-seg.safetensors --visual-prompt prompt.json
    python scripts/export.py --weights weights/yolo26n.safetensors --trt [--fp16]   # 顺带构建 TensorRT engine

--trt：onnx 导出后继续构建序列化 engine（与构建机 GPU/TRT 版本绑定、不跨机移植；换机需重构建）。
--fp16：引擎改为 fp16 构建（需 --trt）：另导半精度图 <stem>.fp16.onnx（model.half()）并据此
构建 <stem>.fp16.engine。每次调用只建一套引擎——`--trt` 建 fp32、`--trt --fp16` 建 fp16，
要两套各跑一次；TRT 11 已移除 FP16 builder flag，精度由 strongly-typed 网络的张量类型决定，
所以 fp16 = 半精度 onnx + strongly-typed 构建；fp32 与 fp16 两套产物文件名不同、可共存。
fp16 是有损的部署取向：先跑 scripts/eval.py 对比精度再决定是否采用（引擎输出统一转回
fp32，下游契约不变）。
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # 仓库根目录入 sys.path

import torch

from model.weights import (arch_has_scale, load_meta, load_weights, provenance_meta,
                           prompt_mode_from_weights, resolve_arch_scale, resolve_imgsz,
                           write_engine_meta, write_onnx_meta)
from model.build import ARCHS, YOLO26_FAMILY, YOLOE26_FAMILY, arch_task, build_model
from utils.logger import get_logger, log_params, setup_logging

setup_logging()
logger = get_logger(__name__)


def _deploy_meta(arch, scale, imgsz, nc, end2end, prompt_mode=None, names=None):
    """导出产物的部署 metadata（ONNX metadata_props / TRT 旁车 JSON 共用）

    与 safetensors metadata 同键名，读取侧统一走 model.weights.load_meta；`end2end` 让
    infer/eval 能校验“图模式 vs --nms”，`names`/`prompt_mode` 让类锁定的 YOLOE text
    图无需原 .safetensors 即可推理/评估。
    """
    meta = {"arch": arch, "scale": scale, "imgsz": int(imgsz), "end2end": bool(end2end)}
    if nc is not None:
        meta["nc"] = int(nc)
    if prompt_mode:
        meta["prompt_mode"] = prompt_mode
    if names:
        meta["names"] = list(names)
    meta.update(provenance_meta())
    return meta




def main():
    parser = argparse.ArgumentParser(description="safetensors -> onnx")
    parser.add_argument("--weights", required=True, help=".safetensors path")
    parser.add_argument("--out", default=None, help="output .onnx path (default runs/export/<weights>.onnx)")
    parser.add_argument("--raw", action="store_true", help="export raw head output (yolo26 only)")
    parser.add_argument("--nms", action="store_true",
                        help="export the o2m+NMS path (yoloe-26 text only; class-locked dense output)")
    parser.add_argument("--dynamic", action="store_true", help="dynamic batch axis (default: fixed batch=1)")
    parser.add_argument("--model", default=None, choices=sorted(ARCHS),
                        help="architecture (default: inferred from the weights filename)")
    parser.add_argument("--scale", default=None,
                        help="model scale (yolo26 family: n/s/m/l/x; default: weights metadata / filename)")
    parser.add_argument("--nc", type=int, default=None,
                        help="class count for models trained on a non-COCO dataset "
                             "(default: weights metadata, else the model yaml's nc)")
    parser.add_argument("--classes", default=None,
                        help="yoloe-26 text mode only: comma-separated class names to lock into the "
                             "exported model (text embeddings are fused into the cls conv; "
                             "the exported onnx rejects new prompts); combine with --visual-prompt "
                             "for mixed prompts")
    parser.add_argument("--text-encoder", dest="text_encoder", default=None,
                        help="MobileCLIP2-B text encoder .safetensors used to fuse text prompts "
                             "(default: mobileclip2_b.safetensors next to --weights)")
    parser.add_argument("--visual-prompt", dest="visual_prompt", default=None,
                        help="yoloe-26 text mode only: JSON with a refer image and bbox/mask prompts; "
                             "the visual embeddings are fused into the cls conv (class-locked); "
                             "combine with --classes for mixed prompts")
    parser.add_argument("--imgsz", type=int, default=None,
                        help="export input size (default: weights metadata, else 640)")
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--trt", action="store_true",
                        help="also build a TensorRT engine (.engine) from the exported onnx (needs tensorrt + CUDA)")
    parser.add_argument("--fp16", action="store_true",
                        help="build the fp16 engine from a half-precision onnx instead of fp32 (requires --trt)")
    args = parser.parse_args()

    arch, scale = resolve_arch_scale(args.weights, args.model, args.scale)
    if arch is None:
        parser.error(f"cannot infer the model from '{args.weights}' — pass --model and/or --scale")
    if arch_has_scale(arch) and scale is None:
        parser.error(f"cannot infer the model scale from '{args.weights}' — pass --scale n/s/m/l/x")
    if arch not in YOLO26_FAMILY and args.raw:
        parser.error(f"--raw is only supported for yolo26 (got model={arch!r})")
    if args.nms and args.raw:
        parser.error("--nms and --raw are mutually exclusive (--raw is the yolo26 dense output; "
                     "--nms is the yoloe-26 text o2m path)")
    if args.fp16 and not args.trt:
        parser.error("--fp16 requires --trt")
    if args.trt and args.dynamic:
        parser.error("--trt requires fixed shapes (drop --dynamic)")
    imgsz = resolve_imgsz(args.weights, args.imgsz)  # CLI > 权重 metadata > 640
    meta = load_meta(args.weights)  # 可选 metadata：训练产物自带 nc（自定义类别免 --nc）
    nc = args.nc if args.nc is not None else meta.get("nc")

    out_path = args.out or str(ROOT / "runs" / "export" / (Path(args.weights).stem + ".onnx"))
    log_params(logger, __file__, weights=args.weights, out=out_path, model=arch, imgsz=imgsz,
               nc=nc if nc is not None else 80, opset=args.opset,
               dynamic=args.dynamic, raw=args.raw, nms=args.nms, trt=args.trt, fp16=args.fp16)

    # YOLOE：prompt 模式经 metadata/文件名解析（pf -> lrpc 布局；text -> 类锁定 fuse 导出）
    prompt_free = None
    text_mode = False
    locked_classes = None
    text_classes = None
    refer_image = visual_prompts = None
    if arch in YOLOE26_FAMILY:
        pm = prompt_mode_from_weights(args.weights, meta)
        prompt_free = True if pm == "pf" else (False if pm == "text" else None)
        if pm == "text":
            text_mode = True
            if meta.get("fused"):
                if args.classes or args.visual_prompt:
                    parser.error("this checkpoint is already class-locked (PE fused) — export it "
                                 "without --classes/--visual-prompt; its prompt names travel in metadata")
            else:
                if args.classes:
                    text_classes = [c.strip() for c in args.classes.split(",")]
                if args.visual_prompt:
                    from utils.visual_prompt import load_visual_prompt

                    try:
                        refer_image, visual_names, visual_prompts = load_visual_prompt(args.visual_prompt)
                    except ValueError as e:
                        parser.error(str(e))
                    if text_classes:
                        dup = sorted(set(text_classes) & set(visual_names))
                        if dup:
                            parser.error(f"text and visual prompts share class names: {dup[:5]} — "
                                         "rename one side or use a single prompt mode")
                        locked_classes = text_classes + visual_names
                    else:
                        locked_classes = visual_names
                else:
                    if not text_classes:
                        # 全量微调产物自带训练时的类名：默认按原词汇导出（仍可用 --classes 覆盖）
                        if meta.get("full_finetune") and meta.get("names"):
                            text_classes = list(meta["names"])
                        else:
                            parser.error("yoloe-26 text-mode export requires --classes or --visual-prompt "
                                         '(e.g. --classes "person, bicycle")')
                    locked_classes = text_classes
    if args.visual_prompt and not text_mode:
        parser.error("--visual-prompt requires yoloe-26 text-mode weights")
    if args.nms and not text_mode:
        parser.error("--nms export is only supported for yoloe-26 text mode "
                     "(prompt-free is E2E only; yolo26 uses --raw)")
    # PE 融合产物：state_dict 是类锁定后的形状（cv3 尾卷积 (c3 -> N)、无 cv4/reprta），
    # 必须按 struct_nc 重建后再 fuse 出同构架构，strict load 才对得上。
    fused = bool(meta.get("fused"))
    struct_nc = meta.get("struct_nc") or nc
    model = build_model(arch, scale, nc=struct_nc, prompt_free=prompt_free)
    model.eval()
    if fused:
        if not text_mode:
            parser.error("fused metadata requires yoloe-26 text-mode weights")
        if args.classes or args.visual_prompt:
            parser.error("this checkpoint is already class-locked (PE fused) — export it without "
                         "--classes/--visual-prompt; its prompt names travel in metadata")
        head = model.model[-1]
        head.fuse_text(torch.zeros(1, int(meta["nc"]), head.embed))  # 只重建结构，值由 strict load 覆盖
    load_weights(model, args.weights, strict=True)
    head = model.model[-1]
    if hasattr(head, "end2end"):  # V3Detect 无此开关（前向即解码）
        head.end2end = not (args.raw or args.nms)
    if hasattr(head, "export"):  # yoloe：export 标志 -> conf=0 静态路径（提案过滤全锚点通过，官方口径）
        head.export = True
    if text_mode and fused:
        locked_classes = list(meta.get("names") or [])
        if len(locked_classes) != int(meta["nc"]):
            parser.error(f"fused metadata names ({len(locked_classes)}) != nc ({meta['nc']}) — "
                         "corrupted checkpoint metadata")
        logger.info(f"prompts already fused: {len(locked_classes)} classes (class-locked checkpoint)")
    elif text_mode:
        vpe = tpe = None
        if args.visual_prompt:
            # 官方视觉 prompt 口径：refer 图三尺度特征 + 1/8 掩码（bbox/mask）-> SAVPE vpe
            import cv2

            from data.preprocess import preprocess
            from utils.visual_prompt import visual_masks

            ref = Path(refer_image)
            if not ref.is_absolute():  # 相对路径按 JSON 所在目录解析（与 infer/eval 同口径）
                ref = Path(args.visual_prompt).resolve().parent / ref
            image = cv2.imread(str(ref))
            if image is None:
                parser.error(f"cannot read visual prompt refer image: {ref}")
            tensor, ratio, pad = preprocess(image, imgsz)
            masks = visual_masks(visual_names, visual_prompts, ratio, pad, imgsz,
                                 image.shape[:2]).unsqueeze(0)
            with torch.no_grad():
                feats = model.forward_feats(tensor)
                vpe = head.get_vpe(feats, masks)
        if text_classes:
            # 官方文本 fuse 口径：tokenize -> MobileCLIP -> get_tpe（reprta+L2）
            from model.text_encoder import load_mobileclip_text
            from model.text_prompt import resolve_text_encoder
            from model.tokenizer import tokenize

            try:
                enc_path = resolve_text_encoder(args.weights, args.text_encoder)
            except (ValueError, FileNotFoundError) as e:
                parser.error(str(e))
            enc = load_mobileclip_text(str(enc_path)).eval()
            with torch.no_grad():
                tokens = tokenize(text_classes)
                emb = torch.cat([enc(chunk) for chunk in tokens.split(80)], dim=0).unsqueeze(0)
                tpe = head.get_tpe(emb)
        # 混合顺序与官方 get_cls_pe 一致：文本在前、视觉在后
        pe = torch.cat([tpe, vpe], dim=1) if (tpe is not None and vpe is not None) else (
            tpe if tpe is not None else vpe)
        head.fuse_text(pe)  # 与文本 fuse 同一路径；fuse 后 nc 锁定
        logger.info(f"prompts fused: {len(locked_classes)} classes"
                    + (" (text + visual)" if tpe is not None and vpe is not None
                       else (" (text)" if tpe is not None else " (visual)")))

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    dummy = torch.randn(1, 3, imgsz, imgsz)
    # 分割头返回 (检测输出, 原型) 二元组 -> 两个输出名；输出名与官方导出一致（output0/output1）
    out_names = ["output0", "output1"] if arch_task(arch) == "segment" else ["output0"]
    # 默认固定 batch=1（边缘工具链偏好固定 shape）；--dynamic 打开动态 batch
    dynamic_axes = {n: {0: "batch"} for n in ["images", *out_names]} if args.dynamic else None
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy,
            out_path,
            input_names=["images"],
            output_names=out_names,
            dynamic_axes=dynamic_axes,
            opset_version=args.opset,
            dynamo=False,  # TorchScript 导出器：权重内嵌单文件；dynamo 默认把权重拆到 .onnx.data
        )
    # 部署 metadata 随图走：类锁定 text 图无需原 .safetensors 也能 infer/eval；
    # end2end 让消费侧校验图模式与 --nms 是否一致。
    deploy_meta = _deploy_meta(
        arch, scale, imgsz,
        len(locked_classes) if text_mode else nc,
        end2end=not (args.raw or args.nms),
        prompt_mode=("text" if text_mode else ("pf" if prompt_free else None)),
        names=(locked_classes if text_mode else (meta.get("names") if prompt_free else None)),
    )
    write_onnx_meta(out_path, deploy_meta)
    if arch in YOLOE26_FAMILY and text_mode:
        kind = ("text + visual prompts" if (args.visual_prompt and text_classes)
                else ("visual prompts" if args.visual_prompt else "text prompts"))
        mode = (("o2m+NMS " if args.nms else "E2E ") + kind +
                " (class-locked) + prototypes")
    elif arch in YOLOE26_FAMILY:
        mode = "E2E prompt-free + prototypes"
    elif arch not in YOLO26_FAMILY:
        mode = "decoded (v3)"
    elif arch_task(arch) == "segment":
        mode = ("raw NMS path + prototypes" if args.raw else "E2E fused + prototypes")
    else:
        mode = "raw NMS path" if args.raw else "E2E fused"
    logger.info(f"exported ({mode}) -> {out_path}")

    if args.trt:
        from utils.engines import build_trt_engine  # 懒加载：非 trt 路径不引入 engine 依赖

        engine_src = Path(out_path)
        if args.fp16:
            # fp16 引擎 = 半精度 onnx + strongly-typed 构建（TRT 11 无 FP16 flag）；独立落盘、
            # 与 fp32 产物共存（<stem>.fp16.onnx -> <stem>.fp16.engine）
            engine_src = Path(out_path).with_name(Path(out_path).stem + ".fp16.onnx")
            with torch.no_grad():
                torch.onnx.export(
                    model.half(),
                    dummy.half(),
                    str(engine_src),
                    input_names=["images"],
                    output_names=out_names,
                    opset_version=args.opset,
                    dynamo=False,
                )
            write_onnx_meta(engine_src, deploy_meta)
            logger.info(f"exported fp16 onnx -> {engine_src}")

        engine_path = engine_src.with_suffix(".engine")
        try:
            build_trt_engine(engine_src, engine_path, fp16=args.fp16)
        except (ImportError, RuntimeError, ValueError) as e:
            parser.error(str(e))
        write_engine_meta(engine_path, deploy_meta)
        mib = Path(engine_path).stat().st_size / 2**20
        logger.info(f"exported TensorRT engine ({'fp16' if args.fp16 else 'fp32'}) -> {engine_path} "
                    f"({mib:.1f} MiB; bound to this GPU + TRT version)")


if __name__ == "__main__":
    main()
