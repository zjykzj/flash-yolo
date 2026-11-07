"""COCO 评估（pt / onnx / trt；YOLO26 E2E/NMS 两路径 + YOLOv3-tiny 解码+NMS）

官方对照（COCO val2017，yolo26n）：
    E2E 路径 mAP 40.1 / NMS 路径 mAP 40.9
yolov3-tiny（darknet 官方权重）：COCO test-dev AP50 = 33.1（本仓库为 pycocotools val2017 口径）。

输出表格与训练侧 val 表同款（11 宽右对齐 + 6 空格缩进），列比训练侧多两项：
    P / R（pycocotools 无此槽位，取 101 点召回网格上 F1 最大点，与训练侧 FastMetrics 同口径）
    AR@100；表尾另打 size buckets（mAP_s/m/l）与 IoU thresholds（mAP50/75）。

用法:
    python scripts/eval.py --weights weights/yolo26n.safetensors --data coco --split val
    python scripts/eval.py --weights weights/yolo26n.safetensors --data /path/to/dataset.yaml
    python scripts/eval.py --weights weights/yolo26n.onnx --engine onnx --data coco --nms
    python scripts/eval.py --weights weights/yolov3-tiny.safetensors --data coco     # v3（档位由架构固定）
    python scripts/eval.py ... --limit 100      # 只跑前 100 张（冒烟）
    python scripts/eval.py ... --summary-only   # 只打 all 行（默认打全部 80 类明细）
    python scripts/eval.py ... --verbose        # 显示第三方库（pycocotools）调试输出

--data 是数据集描述符（名字或 .yaml 路径，见 config/datasets/coco.yaml 的 schema）；
--split 选描述符里的角色键（train/val/test，默认 val）。YOLO txt 数据集同样支持：
GT 由标签现搭 COCO dict，评估口径与 COCO json 完全一致。
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # 仓库根目录入 sys.path

import numpy as np
import torch

from config import __version__
from config.datasets import load_dataset
from config.inference import CONF_THRES, IOU_THRES, MAX_DET
from data.build import build_eval_dataset
from data.scan import scan_summary
from eval import build_evaluator
from eval.coco_evaluator import build_vocab_map
from model.build import ARCHS, YOLO26_FAMILY, YOLOE26_FAMILY, arch_display_name, arch_task
from model.weights import arch_has_scale, load_meta, prompt_mode_from_weights, resolve_arch_scale, resolve_imgsz
from utils.engines import OnnxEngine, PtEngine, TRTEngine, device_label, resolve_device
from utils.logger import attach_file_log, bold, get_logger, log_file_only, log_params, redirect_prints, setup_logging
from utils.paths import increment_path
from utils.progress import ProgressBar

# 控制台立刻可用；文件日志等 run 目录确定后挂（见 attach_file_log）
setup_logging(to_file=False)
logger = get_logger(__name__)

# 表格渲染：6 空格缩进 + 类名左对齐（最长 COCO 类名 "baseball glove" 14 字符，留 1 位余量）
# + 数值右对齐。类名若按数值列宽度右对齐，超宽类名会顶破列宽、把该行数值整体推右。
_NAME_W = 15
# 前两列按表头词留位（Images=6 / Instances=9 字符），否则表头会粘成 "ImagesInstances"
_COL_W = (7, 10, 10, 10, 10, 10, 10)  # Images / Instances / P / R / mAP50 / mAP50-95 / AR@100
def table_header(ar_label="AR@100"):
    return ("      " + f"{'Class':<{_NAME_W}}"
            + "".join(f"{c:>{w}}" for c, w in zip(("Images", "Instances", "P", "R",
                                                   "mAP50", "mAP50-95", ar_label), _COL_W)))


def _fmt_table_row(name, n_img, n_inst, p, r, ap50, ap, ar):
    return ("      " + f"{str(name)[:_NAME_W]:<{_NAME_W}}"
            + "".join(f"{v:>{w}}" for v, w in zip((f"{n_img:d}", f"{n_inst:d}"), _COL_W[:2]))
            + "".join(f"{v:>{w}.4f}" for v, w in zip((p, r, ap50, ap, ar), _COL_W[2:])))


def main():
    parser = argparse.ArgumentParser(description="COCO evaluation")
    parser.add_argument("--weights", required=True, help=".safetensors or .onnx weights")
    parser.add_argument("--data", required=True,
                        help="dataset descriptor: a name in config/datasets/ (local/ wins) or a .yaml path")
    parser.add_argument("--single-cls", dest="single_cls", action="store_true",
                        help="evaluate a single-class checkpoint: remap GT classes to 0 (model nc must be 1)")
    parser.add_argument("--split", default="val", help="role to evaluate (a key in the descriptor: train/val/...)")
    parser.add_argument("--engine", choices=["pt", "onnx", "trt"], default="pt")
    parser.add_argument("--model", default=None, choices=sorted(ARCHS),
                        help="architecture (default: inferred from the weights filename)")
    parser.add_argument("--scale", default=None,
                        help="model scale (yolo26 family: n/s/m/l/x; default: weights metadata / filename)")
    parser.add_argument("--nms", action="store_true",
                        help="use o2m+NMS path (yolo26 family + yoloe-26 text; default E2E NMS-free)")
    parser.add_argument("--agnostic-nms", dest="agnostic_nms", action="store_true",
                        help="class-agnostic NMS (only with --nms; default: per-class suppression)")
    parser.add_argument("--conf", type=float, default=CONF_THRES, help=f"confidence threshold (default {CONF_THRES})")
    parser.add_argument("--iou", type=float, default=IOU_THRES, help=f"NMS IoU threshold (default {IOU_THRES})")
    parser.add_argument("--max-det", type=int, default=MAX_DET)
    parser.add_argument("--device", default=None, help="pt engine device (default auto)")
    parser.add_argument("--imgsz", type=int, default=None,
                        help="model input size (default: weights metadata, else 640)")
    parser.add_argument("--limit", type=int, default=0, help="evaluate first N images only (0=all)")
    parser.add_argument("--eval-style", dest="eval_style", choices=["auto", "coco", "lvis"], default="auto",
                        help="evaluation protocol (default: from the dataset descriptor)")
    parser.add_argument("--eval-max-dets", dest="eval_max_dets", type=int, default=None,
                        help="evaluator maxDets for LVIS (default 300; use 100 to mirror ultralytics)")
    parser.add_argument("--summary-only", action="store_true",
                        help="print the all-classes row only (per-class rows are shown by default)")
    parser.add_argument("--verbose", action="store_true", help="show third-party (pycocotools) debug output")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel("DEBUG")

    try:  # 描述符错误在建 run 目录之前拦下（无 traceback）
        spec = load_dataset(args.data)
    except (ValueError, FileNotFoundError) as e:
        parser.error(str(e))

    # 评估协议：描述符 eval_style（默认 coco）> CLI 显式覆盖；LVIS 协议走 faster_coco_eval
    eval_style = spec.eval_style if args.eval_style == "auto" else args.eval_style
    if eval_style == "lvis" and spec.format != "coco":
        parser.error("LVIS evaluation requires a COCO-format dataset descriptor (eval_style: lvis)")
    if args.eval_max_dets is not None and eval_style != "lvis":
        parser.error("--eval-max-dets is only valid with LVIS evaluation (--eval-style lvis)")
    eval_max_dets = (300 if args.eval_max_dets is None else args.eval_max_dets) if eval_style == "lvis" else 100
    eval_backend = "faster-coco-eval" if eval_style == "lvis" else "pycocotools"

    # 架构/档位从权重文件名推（yolo26s -> yolo26/s；yolov3-tiny -> v3）——提前解析，参数预览要用
    arch, scale = resolve_arch_scale(args.weights, args.model, args.scale)
    if arch is None:
        parser.error(f"cannot infer the model from '{args.weights}' — pass --model and/or --scale")
    if arch_has_scale(arch) and scale is None:
        parser.error(f"cannot infer the model scale from '{args.weights}' — pass --scale n/s/m/l/x")
    # YOLOE prompt mode：text 支持 E2E + o2m/NMS；pf 仅 E2E（raw 列数随提案过滤变化）
    meta = load_meta(args.weights)
    prompt_mode = (prompt_mode_from_weights(args.weights, meta)
                   if arch in YOLOE26_FAMILY else None)
    if args.nms:
        if arch in YOLOE26_FAMILY and prompt_mode != "text":
            parser.error("--nms is only supported for yoloe-26 text mode (prompt-free is E2E only)")
        if arch not in YOLO26_FAMILY and arch not in YOLOE26_FAMILY:
            parser.error(f"--nms is only supported for yolo26 / yoloe-26 text (got model={arch!r})")
    if args.agnostic_nms and not args.nms:
        parser.error("--agnostic-nms only affects the NMS path — pass --nms as well")
    # YOLOE：pf 词汇 / 类锁定 text 图都按名字映射到数据集类别；
    # pt 文本模式直接用数据集类别名做 prompt（类别 id = prompt 序号，无需映射）。
    class_map = None
    text_mode = prompt_mode == "text"
    if arch in YOLOE26_FAMILY and not (text_mode and args.engine == "pt"):
        model_names = meta.get("names")
        if not model_names:
            parser.error(f"{arch} evaluation needs the model class names in the metadata — "
                         "convert the weights (scripts/convert_weights.py) or re-export with metadata")
        try:
            class_map = build_vocab_map(model_names, list(spec.names.values()),
                                        getattr(spec, "vocab_synonyms", None) or {})
        except ValueError as e:
            parser.error(str(e))

    # 输入尺寸：CLI > 权重 metadata > 640（预览行用；engine 构建后以 engine.imgsz 为准，onnx 取图内 shape）
    imgsz = resolve_imgsz(args.weights, args.imgsz)

    run_dir = increment_path(ROOT / "runs" / "val" / "val")  # 提前建：日志/结果同目录，评测中即可跟踪
    attach_file_log(run_dir / "run.log")
    log_params(logger, __file__, weights=args.weights, data=args.data, split=args.split,
               model=arch, engine=args.engine, conf=args.conf, iou=args.iou, max_det=args.max_det,
               imgsz=imgsz, eval_style=eval_style, eval_max_dets=eval_max_dets)

    # ---- 头部（与训练日志同一套五段排版：环境 -> 模型 -> 数据集 -> 评估参数 -> 细节）----
    # ① 环境：设备名不依赖 engine（engine 构建 + 3 次 warmup 约 1s，这行要先落地；onnx 固定 CPU）
    device = (resolve_device(args.device) if args.engine == "pt"
              else (torch.device("cuda") if args.engine == "trt" else torch.device("cpu")))
    logger.info(bold(f"Flash-YOLO {__version__} 🚀 Python {sys.version.split()[0]} · torch {torch.__version__} · {device_label(device)}"))

    # ② 模型（arch/scale 已在参数校验段解析）：pt = 模块树口径 / onnx = 部署口径
    #    （两者的"参数量"不是同一个量，见 engine.summary_line）
    end2end = not args.nms
    graph_end2end = meta.get("end2end")
    if args.engine != "pt" and graph_end2end is not None and graph_end2end != end2end:
        parser.error(f"{args.weights} was exported as {'E2E' if graph_end2end else 'o2m+NMS'} "
                     f"but {'--nms' if args.nms else 'E2E (default)'} was requested — "
                     f"{'remove --nms' if graph_end2end else 'pass --nms'}")
    engine_cls = {"pt": PtEngine, "onnx": OnnxEngine, "trt": TRTEngine}[args.engine]
    kwargs = {"end2end": end2end, "scale": scale, "model": arch, "imgsz": args.imgsz,
              "agnostic_nms": args.agnostic_nms}
    if args.engine == "pt":
        kwargs["device"] = args.device
    else:
        kwargs["text_mode"] = text_mode  # 类锁定 YOLOE text：o2m 输出不做 decode_raw
    with redirect_prints(logger):
        engine = engine_cls(args.weights, **kwargs)
    imgsz = engine.imgsz  # onnx 以图内固定 shape 为准（pt = 上面的解析结果）
    if text_mode:
        locked = args.engine != "pt"
        if not locked:
            engine.set_classes(list(spec.names.values()))
        suffix = ", class-locked" if locked else ""
        mode = ("E2E (text prompts" if end2end else "o2m+NMS (text prompts") + suffix + ")"
    elif arch in YOLOE26_FAMILY:
        mode = "E2E (prompt-free)"
    elif arch in YOLO26_FAMILY:
        mode = "E2E (NMS-free)" if end2end else "o2m+NMS"
    else:
        mode = "decode+NMS"
    logger.info(bold(f"{arch_display_name(arch, scale)} · {engine.summary_line} · "
                     f"{mode} · engine {args.engine}"))

    # ③ 数据集：扫描行与训练日志同语法（静态提示 + 进度条 + `└` 汇总）
    # 注意不包 redirect_prints：提示行是裸 print、进度条写的是构造时捕获的 sys.stdout，
    # 包进去会一起被抓成 DEBUG 日志（INFO 级别下凭空消失）
    if args.single_cls:
        nc_meta = meta.get("nc")
        if nc_meta not in (None, 1):
            parser.error(f"--single-cls expects an nc=1 checkpoint, but {args.weights} declares nc={nc_meta}")
    dataset = build_eval_dataset(spec, args.split, progress=True, task=arch_task(arch),
                                 single_cls=args.single_cls)
    logger.info(scan_summary(dataset))
    # 进度条只走控制台（UI 元素不进 logger），文件日志里另落一行同等信息
    log_file_only(f"{args.split}: {len(dataset)} images · {dataset.n_backgrounds} backgrounds · "
                  f"{dataset.n_missing} missing", name=logger.name)
    n_total = len(dataset) if not args.limit else min(args.limit, len(dataset))

    # ④ 评估参数（区分模型侧 max_det 与评估器 maxDets）
    logger.info(f"eval:  split {args.split} · protocol {eval_style} ({eval_backend}) · "
                f"images {n_total}/{len(dataset)} · conf {args.conf} · iou {args.iou} · "
                f"model max_det {args.max_det} · eval maxDets {eval_max_dets}")

    # ---- 评估循环 ----
    # 任务轴：分割跑两个评估器（bbox + segm），指标口径与官方发布的 Box/Mask 双表一致；
    # 两者各自持有一份结果累积（同一份 dets 分别 update），互不影响。
    heads = ["Box"]
    if arch_task(arch) == "segment":
        heads.append("Mask")
    try:  # 协议/依赖错误给干净的 parser.error（LVIS frequency 缺失、faster-coco-eval 未装）
        with redirect_prints(logger):
            evaluators = [(head, build_evaluator(dataset.gt_source(), nc=len(dataset.names),
                                                 iou_type="segm" if head == "Mask" else "bbox",
                                                 eval_style=eval_style, max_dets=eval_max_dets))
                          for head in heads]
    except (ValueError, ImportError) as e:
        parser.error(str(e))
    bar = ProgressBar(n_total, desc="val")
    t0 = time.monotonic()  # 单调时钟（WSL2 墙钟会跳变，见 utils/progress.py 注释）
    t_window, n_window = t0, 0
    timings = {"load": [], "preprocess": [], "inference": [], "postprocess": []}
    for i in range(n_total):
        t_load = time.monotonic()
        image = dataset.load_image(i)
        timings["load"].append((time.monotonic() - t_load) * 1e3)
        dets, tseg = engine.predict_timed(image, conf_thres=args.conf, iou_thres=args.iou)
        if class_map is not None:  # yoloe pf 词汇 / 类锁定 text：模型类别 -> 数据集类别（只保留数据集里的类）
            keep = np.isin(dets.class_ids, list(class_map))
            dets.class_ids = np.array([class_map[c] for c in dets.class_ids[keep]], dtype=np.int64)
            dets.boxes = dets.boxes[keep]
            dets.scores = dets.scores[keep]
            if dets.masks is not None:
                dets.masks = dets.masks[keep]
        for _head, evaluator in evaluators:
            evaluator.update(dataset.image_id(i), dets)
        for k in ("preprocess", "inference", "postprocess"):
            timings[k].append(tseg[k])
        # 每 100 张与最后一张刷新进度（窗口瞬时速度）
        if (i + 1) % 100 == 0 or i == n_total - 1:
            now = time.monotonic()
            window_speed = (i + 1 - n_window) / max(now - t_window, 1e-6)
            t_window, n_window = now, i + 1
            bar.update(i + 1, window_speed)
        if (i + 1) % 500 == 0:
            # 只落日志文件：控制台此刻被进度条的 \r 行占用，logger.info 会与它串成一行
            log_file_only(f"[{i + 1}/{n_total}] elapsed {time.monotonic() - t0:.1f}s", name=logger.name)
    bar.close()
    elapsed = time.monotonic() - t0

    # ---- 结果 ----
    with redirect_prints(logger):
        metrics_by_head = [(head, ev.compute()) for head, ev in evaluators]
    multi = len(metrics_by_head) > 1  # 分割：Box / Mask 两组表

    console_lines, file_lines, detail_lines = [], [], []
    ar_key = f"AR@{eval_max_dets}"
    header = table_header(ar_key)
    for head, metrics in metrics_by_head:
        n_instances = sum(r[2] for r in metrics["per_class"])
        full_table = [_fmt_table_row("all", metrics["images"], n_instances, metrics["P"], metrics["R"],
                                     metrics["mAP@50"], metrics["mAP@[.5:.95]"], metrics[ar_key])]
        full_table += [_fmt_table_row(*row) for row in metrics["per_class"]]
        if multi:
            console_lines.append(bold(f"      {head} metrics"))
            file_lines.append(f"      {head} metrics")
        console_lines += [header] + (full_table[:1] if args.summary_only else full_table)
        file_lines += [header] + full_table  # 文件始终写全表（记录载体，与终端裁剪无关）
        tag = f" ({head})" if multi else ""
        detail_lines.append(
            f"      {f'Size buckets{tag}':<14} mAP_small {metrics['mAP_small']:.4f} · "
            f"mAP_medium {metrics['mAP_medium']:.4f} · mAP_large {metrics['mAP_large']:.4f}"
            f"   (COCO area <32² / 32²-96² / >96²)")
        detail_lines.append(
            f"      {f'Thresholds{tag}':<14} mAP50 {metrics['mAP@50']:.4f} · "
            f"mAP75 {metrics['mAP@75']:.4f} · mAP50-95 {metrics['mAP@[.5:.95]']:.4f}"
            f"   (IoU 0.50 / 0.75 / 0.50:0.95)")
        if eval_style == "lvis":
            detail_lines.append(
                f"      {f'LVIS freq{tag}':<14} APr {metrics['APr']:.4f} · APc {metrics['APc']:.4f} · "
                f"APf {metrics['APf']:.4f}   (rare / common / frequent)")
    for line in console_lines + detail_lines:
        logger.info(line)
    avg = {k: sum(v) / len(v) for k, v in timings.items()}
    total_ms = elapsed / n_total * 1e3
    # 四段计时不含 evaluator/进度条/循环本身；"other" 按端到端残差算，保证加和恒等于 total
    other_ms = total_ms - sum(avg.values())
    speed_line = (f"Speed: {total_ms:.1f} ms/image = load {avg['load']:.1f} + preprocess {avg['preprocess']:.1f} "
                  f"+ inference {avg['inference']:.1f} + postprocess {avg['postprocess']:.1f} "
                  f"+ other {other_ms:.1f} → {n_total / elapsed:.1f} img/s")
    model_line = (f"       model-only inference {avg['inference']:.1f} ms/image "
                  f"({1e3 / max(avg['inference'], 1e-9):.0f} img/s) at shape (1, 3, {imgsz}, {imgsz})")
    done_line = f"Done: {n_total} images · {elapsed:.1f}s"
    if arch == "yolo26":
        ref_lines = [
            "Reference: published yolo26n = E2E 40.1 / NMS 40.9 (Objects365 pretrain + COCO finetune)",
            "           same pipeline with the official weights = E2E 40.27 / NMS 40.89",
        ]
    elif arch == "yolo26-seg":
        ref_lines = [
            "Reference: official yolo26n-seg = box mAP50-95 0.3975 / mask 0.3386 "
            "(training-time validator; pycocotools numbers above are the project's own baseline)",
        ]
    elif arch in YOLO26_FAMILY:
        ref_lines = ["Reference: Flash-YOLO (this repo's own design) — GFLOPs below yolo26n by design"]
    elif arch in YOLOE26_FAMILY:
        ref = "official yoloe-26-seg" if text_mode else "official yoloe-26-seg-pf"
        if eval_style == "lvis":
            ref_lines = [
                f"Reference: {ref} — LVIS minival (ultralytics docs, n scale): "
                f"text 24.7 / visual 21.9 (Non-E2E); pf 16.6 E2E / 17.7 Non-E2E",
                f"Protocol: official LVIS maxDets=300; ultralytics' current val leaves "
                f"faster-coco-eval at maxDets=100 (use --eval-max-dets 100 to mirror)",
            ]
        else:
            ref_lines = [f"Reference: {ref} — LVIS/COCO numbers per the ultralytics docs"]
    else:
        ref_lines = ["Reference: official darknet yolov3-tiny = 33.1 AP50 (COCO test-dev, 416; val2017 here)"]
    for line in ("", speed_line, model_line, done_line, *ref_lines):
        logger.info(line)

    # ---- 保存到 runs/val/valN（递增目录：完整指标表 + COCO 结果 json）----
    # metrics.txt 始终写全表（含 per-class），与 --summary-only 的终端裁剪无关：文件是记录载体
    lines = [f"val: {args.split} {n_total} images · conf={args.conf} · iou={args.iou} · "
             f"model max_det={args.max_det} · protocol={eval_style} ({eval_backend}) · "
             f"eval maxDets={eval_max_dets}",
             *file_lines, "", *detail_lines, speed_line, model_line, done_line, *ref_lines]
    (run_dir / "metrics.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with open(run_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(evaluators[-1][1].results, f)  # 分割写 Mask 组（含 segmentation），检测即唯一一组
    logger.info(f"Results saved to {run_dir}")


if __name__ == "__main__":
    main()
