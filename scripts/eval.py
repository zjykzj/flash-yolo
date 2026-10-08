"""COCO 评估（pt / onnx；YOLO26 E2E/NMS 两路径 + YOLOv3-tiny 解码+NMS）

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

import torch

from config import __version__
from config.datasets import load_dataset
from config.inference import CONF_THRES, IMGSZ, IOU_THRES, MAX_DET
from data.build import build_eval_dataset
from data.scan import scan_summary
from eval.coco_evaluator import CocoEvaluator
from model.build import ARCHS, arch_display_name
from model.weights import resolve_arch_scale
from utils.engine import OnnxEngine, PtEngine, device_label, resolve_device
from utils.logger import attach_file_log, bold, get_logger, log_file_only, redirect_prints, setup_logging
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
TABLE_HEADER = ("      " + f"{'Class':<{_NAME_W}}"
                + "".join(f"{c:>{w}}" for c, w in zip(("Images", "Instances", "P", "R",
                                                       "mAP50", "mAP50-95", "AR@100"), _COL_W)))


def _fmt_table_row(name, n_img, n_inst, p, r, ap50, ap, ar):
    return ("      " + f"{str(name)[:_NAME_W]:<{_NAME_W}}"
            + "".join(f"{v:>{w}}" for v, w in zip((f"{n_img:d}", f"{n_inst:d}"), _COL_W[:2]))
            + "".join(f"{v:>{w}.4f}" for v, w in zip((p, r, ap50, ap, ar), _COL_W[2:])))


def main():
    parser = argparse.ArgumentParser(description="COCO evaluation")
    parser.add_argument("--weights", required=True, help=".safetensors or .onnx weights")
    parser.add_argument("--data", required=True,
                        help="dataset descriptor: a name in config/datasets/ (local/ wins) or a .yaml path")
    parser.add_argument("--split", default="val", help="role to evaluate (a key in the descriptor: train/val/...)")
    parser.add_argument("--engine", choices=["pt", "onnx"], default="pt")
    parser.add_argument("--model", default=None, choices=sorted(ARCHS),
                        help="architecture (default: inferred from the weights filename)")
    parser.add_argument("--scale", default="n", help="model scale (yolo26 only: n/s/m/l/x)")
    parser.add_argument("--nms", action="store_true", help="use o2m+NMS path (yolo26 only; default E2E NMS-free)")
    parser.add_argument("--conf", type=float, default=CONF_THRES, help=f"confidence threshold (default {CONF_THRES})")
    parser.add_argument("--iou", type=float, default=IOU_THRES, help=f"NMS IoU threshold (default {IOU_THRES})")
    parser.add_argument("--max-det", type=int, default=MAX_DET)
    parser.add_argument("--device", default=None, help="pt engine device (default auto)")
    parser.add_argument("--limit", type=int, default=0, help="evaluate first N images only (0=all)")
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

    run_dir = increment_path(ROOT / "runs" / "val" / "val")  # 提前建：日志/结果同目录，评测中即可跟踪
    attach_file_log(run_dir / "run.log")

    # ---- 头部（与训练日志同一套五段排版：环境 -> 模型 -> 数据集 -> 评估参数 -> 细节）----
    # ① 环境：设备名不依赖 engine（engine 构建 + 3 次 warmup 约 1s，这行要先落地；onnx 固定 CPU）
    device = resolve_device(args.device) if args.engine == "pt" else torch.device("cpu")
    logger.info(bold(f"Flash-YOLO {__version__} 🚀 Python {sys.version.split()[0]} · torch {torch.__version__} · {device_label(device)}"))

    # ② 模型：架构/档位从权重文件名推（yolo26s -> yolo26/s；yolov3-tiny -> v3）；
    #    pt = 模块树口径 / onnx = 部署口径（两者的"参数量"不是同一个量，见 engine.summary_line）
    arch, scale = resolve_arch_scale(args.weights, args.model, args.scale)
    if arch is None:
        parser.error(f"cannot infer the model from '{args.weights}' — pass --model and/or --scale")
    if arch == "yolo26" and scale is None:
        parser.error(f"cannot infer the model scale from '{args.weights}' — pass --scale n/s/m/l/x")
    if arch != "yolo26" and args.nms:
        parser.error(f"--nms is only supported for yolo26 (got model={arch!r})")
    end2end = not args.nms
    engine_cls = OnnxEngine if args.engine == "onnx" else PtEngine
    kwargs = {"end2end": end2end, "scale": scale, "model": arch}
    if args.engine == "pt":
        kwargs["device"] = args.device
    with redirect_prints(logger):
        engine = engine_cls(args.weights, **kwargs)
    mode = ("E2E (NMS-free)" if end2end else "o2m+NMS") if arch == "yolo26" else "decode+NMS"
    logger.info(bold(f"{arch_display_name(arch, scale)} · {engine.summary_line} · "
                     f"{mode} · engine {args.engine}"))

    # ③ 数据集：扫描行与训练日志同语法（静态提示 + 进度条 + `└` 汇总）
    # 注意不包 redirect_prints：提示行是裸 print、进度条写的是构造时捕获的 sys.stdout，
    # 包进去会一起被抓成 DEBUG 日志（INFO 级别下凭空消失）
    dataset = build_eval_dataset(spec, args.split, progress=True)
    logger.info(scan_summary(dataset))
    # 进度条只走控制台（UI 元素不进 logger），文件日志里另落一行同等信息
    log_file_only(f"{args.split}: {len(dataset)} images · {dataset.n_backgrounds} backgrounds · "
                  f"{dataset.n_missing} missing", name=logger.name)
    n_total = len(dataset) if not args.limit else min(args.limit, len(dataset))

    # ④ 评估参数
    logger.info(f"eval:  split {args.split} · images {n_total}/{len(dataset)} · "
                f"conf {args.conf} · iou {args.iou} · max_det {args.max_det}")

    # ---- 评估循环 ----
    with redirect_prints(logger):
        evaluator = CocoEvaluator(dataset.gt_source(), nc=len(dataset.names))
    bar = ProgressBar(n_total, desc="val")
    t0 = time.monotonic()  # 单调时钟（WSL2 墙钟会跳变，见 utils/progress.py 注释）
    t_window, n_window = t0, 0
    timings = {"load": [], "preprocess": [], "inference": [], "postprocess": []}
    for i in range(n_total):
        t_load = time.monotonic()
        image = dataset.load_image(i)
        timings["load"].append((time.monotonic() - t_load) * 1e3)
        dets, tseg = engine.predict_timed(image, conf_thres=args.conf, iou_thres=args.iou)
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
        metrics = evaluator.compute()

    n_instances = sum(r[2] for r in metrics["per_class"])
    full_table = [_fmt_table_row("all", metrics["images"], n_instances, metrics["P"], metrics["R"],
                                 metrics["mAP@50"], metrics["mAP@[.5:.95]"], metrics["AR@100"])]
    full_table += [_fmt_table_row(*row) for row in metrics["per_class"]]
    for line in [TABLE_HEADER] + (full_table[:1] if args.summary_only else full_table):
        logger.info(line)

    size_line = (f"      {'Size buckets':<14} mAP_small {metrics['mAP_small']:.4f} · "
                 f"mAP_medium {metrics['mAP_medium']:.4f} · mAP_large {metrics['mAP_large']:.4f}"
                 f"   (COCO area <32² / 32²-96² / >96²)")
    thr_line = (f"      {'Thresholds':<14} mAP50 {metrics['mAP@50']:.4f} · mAP75 {metrics['mAP@75']:.4f} · "
                f"mAP50-95 {metrics['mAP@[.5:.95]']:.4f}   (IoU 0.50 / 0.75 / 0.50:0.95)")
    avg = {k: sum(v) / len(v) for k, v in timings.items()}
    total_ms = elapsed / n_total * 1e3
    # 四段计时不含 evaluator/进度条/循环本身；"other" 按端到端残差算，保证加和恒等于 total
    other_ms = total_ms - sum(avg.values())
    speed_line = (f"Speed: {total_ms:.1f} ms/image = load {avg['load']:.1f} + preprocess {avg['preprocess']:.1f} "
                  f"+ inference {avg['inference']:.1f} + postprocess {avg['postprocess']:.1f} "
                  f"+ other {other_ms:.1f} → {n_total / elapsed:.1f} img/s")
    model_line = (f"       model-only inference {avg['inference']:.1f} ms/image "
                  f"({1e3 / max(avg['inference'], 1e-9):.0f} img/s) at shape (1, 3, {IMGSZ}, {IMGSZ})")
    done_line = f"Done: {n_total} images · {elapsed:.1f}s"
    if arch == "yolo26":
        ref_lines = [
            "Reference: published yolo26n = E2E 40.1 / NMS 40.9 (Objects365 pretrain + COCO finetune)",
            "           same pipeline with the official weights = E2E 40.27 / NMS 40.89",
        ]
    else:
        ref_lines = ["Reference: official darknet yolov3-tiny = 33.1 AP50 (COCO test-dev, 416; val2017 here)"]
    for line in ("", size_line, thr_line, speed_line, model_line, done_line, *ref_lines):
        logger.info(line)

    # ---- 保存到 runs/val/valN（递增目录：完整指标表 + COCO 结果 json）----
    # metrics.txt 始终写全表（含 per-class），与 --summary-only 的终端裁剪无关：文件是记录载体
    lines = [f"val: {args.split} {n_total} images · conf={args.conf} · iou={args.iou} · max_det={args.max_det}",
             TABLE_HEADER, *full_table, "", size_line, thr_line, speed_line, model_line, done_line, *ref_lines]
    (run_dir / "metrics.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with open(run_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(evaluator.results, f)
    logger.info(f"Results saved to {run_dir}")


if __name__ == "__main__":
    main()
