"""单图 / 目录推理 + 可视化（pt / onnx）

用法:
    python scripts/infer.py --weights weights/yolo26n.safetensors --image assets/bus.jpg
    python scripts/infer.py --weights yolo26n.onnx --engine onnx --image assets/     # 目录批量
    python scripts/infer.py --weights weights/yolo26n.safetensors --image bus.jpg --nms  # o2m+NMS 路径
"""

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # 仓库根目录入 sys.path

import cv2

from config import __version__
from config.defaults import COCO_NAMES, IMGSZ
from utils.engine import OnnxEngine, PtEngine
from utils.logger import attach_file_log, bold, get_logger, setup_logging
from utils.paths import increment_path
from utils.visualize import draw_detections

# 控制台立刻可用；文件日志等 run 目录确定后挂（见 attach_file_log）
setup_logging(to_file=False)
logger = get_logger(__name__)

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _collect_images(source):
    """单图路径或目录 -> 图片路径列表（目录模式按文件名排序）"""
    p = Path(source)
    if p.is_dir():
        return sorted(f for f in p.iterdir() if f.suffix.lower() in _IMG_EXTS)
    return [p]


def _summary(image_path, image, dets, pipeline_ms, idx, total):
    """ultralytics predict 风格摘要：image i/n <path>: WxH <类别统计>, <管线耗时>ms"""
    s = f"image {idx}/{total} {image_path}: {image.shape[1]}x{image.shape[0]} "
    counts = Counter(int(c) for c in dets.class_ids)
    if counts:
        s += ", ".join(f"{n} {COCO_NAMES.get(cid, cid)}{'s' * (n > 1)}" for cid, n in sorted(counts.items())) + ", "
    return s + f"{pipeline_ms:.1f}ms"


def main():
    parser = argparse.ArgumentParser(description="YOLO26 single-image / directory inference")
    parser.add_argument("--weights", required=True, help=".safetensors or .onnx weights path")
    parser.add_argument("--image", required=True, help="input image path or directory")
    parser.add_argument("--engine", choices=["pt", "onnx"], default="pt")
    parser.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--nms", action="store_true", help="use o2m+NMS path (default E2E NMS-free)")
    parser.add_argument(
        "--output", default=None,
        help="output path: single image file, or directory for directory mode (default runs/predict/predictN/)",
    )
    parser.add_argument("--device", default=None, help="pt engine device (default auto)")
    args = parser.parse_args()

    end2end = not args.nms
    engine_cls = OnnxEngine if args.engine == "onnx" else PtEngine
    kwargs = {"end2end": end2end}
    if args.engine == "pt":
        kwargs["device"] = args.device
    engine = engine_cls(args.weights, **kwargs)

    images = _collect_images(args.image)
    if not images:
        raise FileNotFoundError(f"no images found: {args.image}")
    multi = len(images) > 1 or Path(args.image).is_dir()

    # 结果保存：默认 runs/predict/predictN/（递增）；--output 可显式指定文件（单图）或目录（多图）
    if args.output:
        run_dir = Path(args.output).resolve() if multi else Path(args.output).resolve().parent
    else:
        run_dir = increment_path(ROOT / "runs" / "predict" / "predict")
    run_dir.mkdir(parents=True, exist_ok=True)
    attach_file_log(run_dir / "run.log")

    logger.info(bold(f"Flash-YOLO {__version__} · YOLO26n · {'E2E (NMS-free)' if end2end else 'o2m+NMS'} · engine {args.engine}"))

    t_start = time.perf_counter()  # 端到端计时（含图像读/写 I/O）
    stage_sums = {"preprocess": 0.0, "inference": 0.0, "postprocess": 0.0}
    n_done = 0
    labels_dir = run_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    for idx, path in enumerate(images, 1):
        image = cv2.imread(str(path))
        if image is None:
            logger.warning(f"skip unreadable image: {path}")
            continue
        n_done += 1
        dets, tseg = engine.predict_timed(image, conf_thres=args.conf)
        for k in stage_sums:
            stage_sums[k] += tseg[k]

        out_path = run_dir / path.name
        draw_detections(image, dets, COCO_NAMES, save_path=str(out_path))

        # YOLO 格式标签：labels/<同名>.txt，每行 cls xc yc w h（归一化 0-1）
        h_img, w_img = image.shape[:2]
        with open(labels_dir / (path.stem + ".txt"), "w", encoding="utf-8") as f:
            for box, cid in zip(dets.boxes, dets.class_ids):
                x1, y1, x2, y2 = box
                xc, yc = (x1 + x2) / 2 / w_img, (y1 + y2) / 2 / h_img
                w, h = (x2 - x1) / w_img, (y2 - y1) / h_img
                f.write(f"{int(cid)} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

        pipeline_ms = sum(tseg.values())
        logger.info(_summary(path, image, dets, pipeline_ms, idx, len(images)))

    if n_done == 0:
        logger.warning("no valid images processed")
        return
    elapsed_ms = (time.perf_counter() - t_start) * 1e3

    # ---- 汇总（单图与多图同格式）----
    avg = {k: v / n_done for k, v in stage_sums.items()}
    logger.info(
        f"Speed: {avg['preprocess']:.1f}ms preprocess, {avg['inference']:.1f}ms inference, "
        f"{avg['postprocess']:.1f}ms postprocess per image at shape (1, 3, {IMGSZ}, {IMGSZ})"
    )
    logger.info(f"Total: {elapsed_ms:.1f}ms end-to-end for {n_done} image(s) (incl. image read + save)")
    logger.info(f"Visualization -> {run_dir}")
    logger.info(f"Labels        -> {labels_dir}")


if __name__ == "__main__":
    main()
