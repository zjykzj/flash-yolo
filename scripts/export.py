"""pt (safetensors) -> onnx 导出

yolo26：默认导出 E2E 融合图，单输出 output0 (B, 300, 6)，解码与两阶段 top-k 全部在图内；
--raw 导出原始头输出 (B, 4+nc, 8400)，配合 NMS 路径在外部解码（utils/postprocess.py）。
yolo26-seg：双输出——output0 在检测输出末维追加 nm 个 mask 系数（E2E: (B, 300, 6+nm)；
raw: (B, 4+nc+nm, 8400)），output1 是原型 (B, nm, H/4, W/4)；掩码在外部由
utils/postprocess.process_mask 解码（与官方 val 逐像素一致）。
yolov3-tiny：单输出解码 (B, NA, 5+nc)——obj×cls 与按类 NMS 在外部（utils/postprocess.v3_detections），
--raw 不适用。
默认 batch 固定为 1（边缘工具链偏好固定 shape），--dynamic 打开动态 batch。

用法:
    python scripts/export.py --weights weights/yolo26n.safetensors --out weights/yolo26n.onnx
    python scripts/export.py --weights weights/yolo26n.safetensors --out weights/yolo26n_raw.onnx --raw
    python scripts/export.py --weights weights/yolov3-tiny.safetensors --model yolov3-tiny
    python scripts/export.py --weights weights/yolo26n.safetensors --trt [--fp16]   # 顺带构建 TensorRT engine

--trt：onnx 导出后继续构建序列化 engine（与构建机 GPU/TRT 版本绑定、不跨机移植；换机需重构建）。
--fp16：额外导出半精度图 <stem>.fp16.onnx（model.half()）并据此构建 fp16 引擎 <stem>.fp16.engine——
TRT 11 已移除 FP16 builder flag，精度由 strongly-typed 网络的张量类型决定，所以 fp16 = 半精度 onnx
+ strongly-typed 构建；fp32 与 fp16 两套产物文件名不同、可共存。fp16 是有损的部署取向：先跑
scripts/eval.py 对比精度再决定是否采用（引擎输出统一转回 fp32，下游契约不变）。
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # 仓库根目录入 sys.path

import torch

from model.weights import arch_has_scale, load_meta, load_weights, resolve_arch_scale, resolve_imgsz
from model.build import ARCHS, YOLO26_FAMILY, arch_task, build_model
from utils.logger import get_logger, log_params, setup_logging

setup_logging()
logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="safetensors -> onnx")
    parser.add_argument("--weights", required=True, help=".safetensors path")
    parser.add_argument("--out", default=None, help="output .onnx path (default runs/export/<weights>.onnx)")
    parser.add_argument("--raw", action="store_true", help="export raw head output (yolo26 only)")
    parser.add_argument("--dynamic", action="store_true", help="dynamic batch axis (default: fixed batch=1)")
    parser.add_argument("--model", default=None, choices=sorted(ARCHS),
                        help="architecture (default: inferred from the weights filename)")
    parser.add_argument("--scale", default=None,
                        help="model scale (yolo26 family: n/s/m/l/x; default: weights metadata / filename)")
    parser.add_argument("--nc", type=int, default=None,
                        help="class count for models trained on a non-COCO dataset "
                             "(default: weights metadata, else the model yaml's nc)")
    parser.add_argument("--imgsz", type=int, default=None,
                        help="export input size (default: weights metadata, else 640)")
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--trt", action="store_true",
                        help="also build a TensorRT engine (.engine) from the exported onnx (needs tensorrt + CUDA)")
    parser.add_argument("--fp16", action="store_true",
                        help="fp16 deployment: also export a half-precision onnx and build the engine from it "
                             "(requires --trt)")
    args = parser.parse_args()

    arch, scale = resolve_arch_scale(args.weights, args.model, args.scale)
    if arch is None:
        parser.error(f"cannot infer the model from '{args.weights}' — pass --model and/or --scale")
    if arch_has_scale(arch) and scale is None:
        parser.error(f"cannot infer the model scale from '{args.weights}' — pass --scale n/s/m/l/x")
    if arch not in YOLO26_FAMILY and args.raw:
        parser.error(f"--raw is only supported for yolo26 (got model={arch!r})")
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
               dynamic=args.dynamic, raw=args.raw, trt=args.trt, fp16=args.fp16)

    model = build_model(arch, scale, nc=nc)
    load_weights(model, args.weights, strict=True)
    head = model.model[-1]
    if hasattr(head, "end2end"):  # V3Detect 无此开关（前向即解码）
        head.end2end = not args.raw
    model.eval()

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
    if arch not in YOLO26_FAMILY:
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
            logger.info(f"exported fp16 onnx -> {engine_src}")

        engine_path = engine_src.with_suffix(".engine")
        try:
            build_trt_engine(engine_src, engine_path, fp16=args.fp16)
        except (ImportError, RuntimeError, ValueError) as e:
            parser.error(str(e))
        mib = Path(engine_path).stat().st_size / 2**20
        logger.info(f"exported TensorRT engine ({'fp16' if args.fp16 else 'fp32'}) -> {engine_path} "
                    f"({mib:.1f} MiB; bound to this GPU + TRT version)")


if __name__ == "__main__":
    main()
