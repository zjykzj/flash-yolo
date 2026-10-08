"""单图 / 目录推理 + 可视化（pt / onnx；YOLO26 / YOLOv3-tiny）

用法:
    python scripts/infer.py --weights weights/yolo26n.safetensors --image assets/bus.jpg
    python scripts/infer.py --weights yolo26n.onnx --engine onnx --image assets/     # 目录批量
    python scripts/infer.py --weights weights/yolo26n.safetensors --image bus.jpg --nms  # o2m+NMS 路径
    python scripts/infer.py --weights weights/yolov3-tiny.safetensors --image bus.jpg  # v3（档位由架构固定）
"""

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # 仓库根目录入 sys.path

import cv2
import torch

from config import __version__
from config.datasets import load_dataset, load_names
from config.inference import IMGSZ
from model.build import ARCHS, arch_display_name
from model.weights import resolve_arch_scale
from utils.engine import OnnxEngine, PtEngine, device_label, resolve_device
from utils.logger import attach_file_log, bold, get_logger, log_params, setup_logging
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


def _summary(image_path, image, dets, pipeline_ms, idx, total, names):
    """ultralytics predict 风格摘要：image i/n <path>: WxH <类别统计>, <管线耗时>ms"""
    s = f"image {idx}/{total} {image_path}: {image.shape[1]}x{image.shape[0]} "
    counts = Counter(int(c) for c in dets.class_ids)
    if counts:
        s += ", ".join(f"{n} {names.get(cid, cid)}{'s' * (n > 1)}" for cid, n in sorted(counts.items())) + ", "
    return s + f"{pipeline_ms:.1f}ms"


def main():
    parser = argparse.ArgumentParser(description="single-image / directory inference (YOLO26 / YOLOv3-tiny)")
    parser.add_argument("--weights", required=True, help=".safetensors or .onnx weights path")
    parser.add_argument("--image", required=True, help="input image path or directory")
    parser.add_argument("--engine", choices=["pt", "onnx"], default="pt")
    parser.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--nms", action="store_true", help="use o2m+NMS path (yolo26 only)")
    parser.add_argument("--model", default=None, choices=sorted(ARCHS),
                        help="architecture (default: inferred from the weights filename)")
    parser.add_argument(
        "--output", default=None,
        help="output path: single image file, or directory for directory mode (default runs/predict/predictN/)",
    )
    parser.add_argument("--device", default=None, help="pt engine device (default auto)")
    parser.add_argument("--scale", default=None,
                        help="model scale (yolo26 only: n/s/m/l/x; default: inferred from the weights filename)")
    parser.add_argument("--data", default=None,
                        help="dataset descriptor (name or .yaml): sets nc and label names for models "
                             "trained on a non-COCO dataset (default: COCO, nc from the model yaml)")
    args = parser.parse_args()

    names, nc = load_names(), None  # 推理无数据集上下文：默认按 COCO 类名/80 类构建
    if args.data:
        try:
            spec = load_dataset(args.data)
        except (ValueError, FileNotFoundError) as e:
            parser.error(str(e))
        names, nc = spec.names, spec.nc

    # 输入清单与 run 目录先确定：日志挂到 run 目录上（engine 构建失败的报错也才进得了 run.log）
    images = _collect_images(args.image)
    if not images:
        raise FileNotFoundError(f"no images found: {args.image}")
    multi = len(images) > 1 or Path(args.image).is_dir()

    # 架构/档位从权重文件名推（yolo26s -> yolo26/s；yolov3-tiny -> v3）——提前解析，参数预览要用
    arch, scale = resolve_arch_scale(args.weights, args.model, args.scale)
    if arch is None:
        parser.error(f"cannot infer the model from '{args.weights}' — pass --model and/or --scale")
    if arch == "yolo26" and scale is None:
        parser.error(f"cannot infer the model scale from '{args.weights}' — pass --scale n/s/m/l/x")
    if arch != "yolo26" and args.nms:
        parser.error(f"--nms is only supported for yolo26 (got model={arch!r})")

    # 结果保存：默认 runs/predict/predictN/（递增）；--output 可显式指定文件（单图）或目录（多图）
    if args.output:
        run_dir = Path(args.output).resolve() if multi else Path(args.output).resolve().parent
    else:
        run_dir = increment_path(ROOT / "runs" / "predict" / "predict")
    run_dir.mkdir(parents=True, exist_ok=True)
    attach_file_log(run_dir / "run.log")
    log_params(logger, __file__, weights=args.weights, image=args.image, model=arch,
               engine=args.engine, conf=args.conf, imgsz=IMGSZ)

    # ---- 头部（与训练/评估同一套五段排版：环境 -> 模型 -> 输入/参数 -> 细节）----
    # ① 环境：engine 构建之前（onnx 后端固定 CPU，见 OnnxEngine）
    device = resolve_device(args.device) if args.engine == "pt" else "cpu"
    logger.info(bold(f"Flash-YOLO {__version__} 🚀 Python {sys.version.split()[0]} · torch {torch.__version__} · {device_label(device)}"))

    # ② 模型（arch/scale 已在前段解析）
    end2end = not args.nms
    engine_cls = OnnxEngine if args.engine == "onnx" else PtEngine
    kwargs = {"end2end": end2end, "scale": scale, "model": arch}
    if args.engine == "pt":
        kwargs["device"] = args.device
        kwargs["nc"] = nc  # onnx 图内置类别数，只有 pt 引擎需要重建头
    engine = engine_cls(args.weights, **kwargs)
    mode = ("E2E (NMS-free)" if end2end else "o2m+NMS") if arch == "yolo26" else "decode+NMS"
    logger.info(bold(f"{arch_display_name(arch, scale)} · {engine.summary_line} · "
                     f"{mode} · engine {args.engine}"))

    # ③ 输入清单 + 本次任务参数
    logger.info(f"infer: {args.image} · {len(images)} image{'s' if len(images) > 1 else ''} · "
                f"conf {args.conf} · imgsz {IMGSZ}")

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
        draw_detections(image, dets, names, save_path=str(out_path))

        # YOLO 格式标签：labels/<同名>.txt，每行 cls xc yc w h（归一化 0-1）
        h_img, w_img = image.shape[:2]
        with open(labels_dir / (path.stem + ".txt"), "w", encoding="utf-8") as f:
            for box, cid in zip(dets.boxes, dets.class_ids):
                x1, y1, x2, y2 = box
                xc, yc = (x1 + x2) / 2 / w_img, (y1 + y2) / 2 / h_img
                w, h = (x2 - x1) / w_img, (y2 - y1) / h_img
                f.write(f"{int(cid)} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

        pipeline_ms = sum(tseg.values())
        logger.info(_summary(path, image, dets, pipeline_ms, idx, len(images), names))

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
