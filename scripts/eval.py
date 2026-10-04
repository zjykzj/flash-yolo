"""COCO 评估（pt / onnx，E2E 与 NMS 两条路径）

官方对照（COCO val2017，yolo26n）：
    E2E 路径 mAP 40.1 / NMS 路径 mAP 40.9

用法:
    python scripts/eval.py --weights weights/yolo26n.safetensors --data /path/to/coco --split val2017
    python scripts/eval.py --weights weights/yolo26n.onnx --engine onnx --data /path/to/coco --nms
    python scripts/eval.py ... --limit 100      # 只跑前 100 张（冒烟）
    python scripts/eval.py ... --per-class      # 展开 80 类明细表
    python scripts/eval.py ... --verbose        # 显示第三方库（pycocotools）调试输出
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
from config.defaults import CONF_THRES, IMGSZ, IOU_THRES, MAX_DET
from data.coco import CocoDataset
from eval.coco_evaluator import CocoEvaluator
from utils.engine import OnnxEngine, PtEngine
from utils.logger import bold, get_logger, redirect_prints, setup_logging
from utils.paths import increment_path
from utils.progress import ProgressBar

setup_logging()
logger = get_logger(__name__)


def _device_name(engine):
    device = getattr(engine, "device", None)
    if device is not None and device.type == "cuda":
        return f"CUDA {torch.cuda.get_device_name(device)}"
    return "CPU"


def main():
    parser = argparse.ArgumentParser(description="COCO evaluation")
    parser.add_argument("--weights", required=True, help=".safetensors or .onnx weights")
    parser.add_argument("--data", required=True, help="COCO data root (containing annotations/ and val2017/)")
    parser.add_argument("--split", default="val2017")
    parser.add_argument("--engine", choices=["pt", "onnx"], default="pt")
    parser.add_argument("--scale", default="n", help="model scale (n/s/m/l/x, used by pt engine)")
    parser.add_argument("--nms", action="store_true", help="use o2m+NMS path (default E2E NMS-free)")
    parser.add_argument("--conf", type=float, default=CONF_THRES, help=f"confidence threshold (default {CONF_THRES})")
    parser.add_argument("--iou", type=float, default=IOU_THRES, help=f"NMS IoU threshold (default {IOU_THRES})")
    parser.add_argument("--max-det", type=int, default=MAX_DET)
    parser.add_argument("--device", default=None, help="pt engine device (default auto)")
    parser.add_argument("--limit", type=int, default=0, help="evaluate first N images only (0=all)")
    parser.add_argument("--per-class", action="store_true", help="show per-class table")
    parser.add_argument("--verbose", action="store_true", help="show third-party (pycocotools) debug output")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel("DEBUG")

    with redirect_prints(logger):  # pycocotools 的裸 print -> DEBUG
        dataset = CocoDataset(args.data, args.split)

    end2end = not args.nms
    engine_cls = OnnxEngine if args.engine == "onnx" else PtEngine
    kwargs = {"end2end": end2end, "scale": args.scale}
    if args.engine == "pt":
        kwargs["device"] = args.device
    with redirect_prints(logger):
        engine = engine_cls(args.weights, **kwargs)

    n_total = len(dataset) if not args.limit else min(args.limit, len(dataset))

    # ---- 头部 ----
    layers, n_params = engine.summary
    model_line = f"YOLO26{args.scale}"
    if layers:
        model_line += f" · {layers} layers · {n_params:,} params"
    else:
        model_line += f" · {n_params:,} params"
    model_line += f" · {('E2E (NMS-free)' if end2end else 'o2m+NMS')} · engine {args.engine}"
    logger.info(bold(f"Flash-YOLO {__version__} 🚀 Python {sys.version.split()[0]} · torch {torch.__version__} · {_device_name(engine)}"))
    logger.info(bold(model_line))
    logger.info(f"val: {args.split} {n_total} images · conf={args.conf} · iou={args.iou}")

    # ---- 评估循环 ----
    with redirect_prints(logger):
        evaluator = CocoEvaluator(dataset.ann_file)
    bar = ProgressBar(n_total, desc="val")
    t0 = time.monotonic()  # 单调时钟（WSL2 墙钟会跳变，见 utils/progress.py 注释）
    t_window, n_window = t0, 0
    timings = {"preprocess": [], "inference": [], "postprocess": []}
    for i in range(n_total):
        image = dataset.load_image(i)
        dets, tseg = engine.predict_timed(image, conf_thres=args.conf, iou_thres=args.iou)
        evaluator.update(dataset.image_id(i), dets)
        for k in timings:
            timings[k].append(tseg[k])
        # 每 100 张与最后一张刷新进度（窗口瞬时速度）
        if (i + 1) % 100 == 0 or i == n_total - 1:
            now = time.monotonic()
            window_speed = (i + 1 - n_window) / max(now - t_window, 1e-6)
            t_window, n_window = now, i + 1
            bar.update(i + 1, window_speed)
        if (i + 1) % 500 == 0:
            logger.info(f"[{i + 1}/{n_total}] elapsed {time.monotonic() - t0:.1f}s")
    bar.close()
    elapsed = time.monotonic() - t0

    # ---- 结果 ----
    with redirect_prints(logger):
        metrics = evaluator.compute()

    header = f"{'Class':>20} {'Images':>7} {'Instances':>10} {'mAP50':>8} {'mAP50-95':>9} {'AR':>7}"
    n_instances = sum(r[2] for r in metrics["per_class"])
    table = [
        header,
        f"{'all':>20} {metrics['images']:>7} {n_instances:>10} "
        f"{metrics['mAP@50']:>8.4f} {metrics['mAP@[.5:.95]']:>9.4f} {metrics['AR@100']:>7.4f}",
    ]
    if args.per_class:
        table += [
            f"{name:>20} {n_imgs:>7} {n_inst:>10} {ap50:>8.4f} {ap:>9.4f} {ar:>7.4f}"
            for name, n_imgs, n_inst, ap50, ap, ar in metrics["per_class"]
        ]
    for line in table:
        logger.info(line)

    avg = {k: sum(v) / len(v) for k, v in timings.items()}
    speed_line = (
        f"Speed: {avg['preprocess']:.1f}ms preprocess, {avg['inference']:.1f}ms inference, "
        f"{avg['postprocess']:.1f}ms postprocess per image at shape (1, 3, {IMGSZ}, {IMGSZ})"
    )
    done_line = f"Done: {n_total} images · {elapsed:.1f}s · avg {n_total / elapsed:.1f} img/s"
    logger.info(speed_line)
    logger.info(done_line)
    logger.info("Official reference (yolo26n): E2E 40.1 / NMS path 40.9")

    # ---- 保存到 runs/val/valN（递增目录：指标表 + COCO 结果 json）----
    run_dir = increment_path(ROOT / "runs" / "val" / "val")
    lines = [f"val: {args.split} {n_total} images · conf={args.conf} · iou={args.iou}", *table, speed_line, done_line]
    (run_dir / "metrics.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with open(run_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(evaluator.results, f)
    logger.info(f"Results saved to {run_dir}")


if __name__ == "__main__":
    main()
