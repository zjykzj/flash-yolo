"""COCO json -> YOLO-seg txt（class + 归一化多边形）

dataflow-cv 的 `convert coco2yolo` 目前即使标注带 segmentation 也优先写 bbox（5 列）；
本脚本用本仓库 `data/segments.py` 的官方 merge_multi_segment 口径生成多边形行，保证
YOLO-seg 与 COCO 训练路径的多边形几何一致。RLE/缺失 segmentation 的标注写 5 列 bbox 行，
加载器按矩形回退（与 COCO 侧同口径）；iscrowd=1 剔除（YOLO 无 crowd）。

输出：`<dst>/<image_id>.txt`，每行 `cls x1 y1 x2 y2 ...`（归一化 0-1）或 `cls xc yc w h`。
数据集描述符仍由 `config/datasets` 提供（names 顺序 = COCO category_id 升序）。

用法:
    python scripts/convert_coco_seg.py \\
        --src dataset/coco/annotations/instances_train2017.json \\
        --dst dataset/coco/labels/train2017
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from data.segments import annotation_polygon
from utils.logger import get_logger, log_params, setup_logging

setup_logging()
logger = get_logger(__name__)


def convert(src, dst):
    """返回统计 dict：images / polygons / bbox_fallback / skipped / crowd"""
    data = json.loads(Path(src).read_text())
    cats = sorted(data["categories"], key=lambda c: c["id"])
    cat2idx = {c["id"]: i for i, c in enumerate(cats)}
    images = {img["id"]: img for img in data["images"]}
    per_image = {}
    n_poly = n_bbox = n_skip = n_crowd = 0
    for ann in data["annotations"]:
        if ann.get("iscrowd", 0):
            n_crowd += 1
            continue
        info = images.get(ann["image_id"])
        cls = cat2idx.get(ann.get("category_id"))
        if info is None or cls is None:
            n_skip += 1
            continue
        w, h = float(info["width"]), float(info["height"])
        if w <= 0 or h <= 0:
            n_skip += 1
            continue
        poly, fallback = annotation_polygon(ann)
        if poly is None:
            n_skip += 1
            continue
        if fallback:
            bbox = ann.get("bbox")
            if not bbox or len(bbox) < 4:
                n_skip += 1
                continue
            x, y, bw, bh = (float(v) for v in bbox[:4])
            line = (f"{cls} {(x + bw / 2) / w:.6f} {(y + bh / 2) / h:.6f} "
                    f"{bw / w:.6f} {bh / h:.6f}")
            n_bbox += 1
        else:
            pts = poly / np.array([w, h], np.float32)
            line = f"{cls} " + " ".join(f"{v:.6f}" for v in pts.reshape(-1))
            n_poly += 1
        per_image.setdefault(ann["image_id"], []).append(line)

    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    for iid, lines in per_image.items():
        (dst / f"{iid}.txt").write_text("\n".join(lines), encoding="utf-8")
    return {"images": len(per_image), "polygons": n_poly, "bbox_fallback": n_bbox,
            "skipped": n_skip, "crowd": n_crowd}


def main():
    parser = argparse.ArgumentParser(description="COCO json -> YOLO-seg txt labels")
    parser.add_argument("--src", required=True, help="COCO instances_*.json")
    parser.add_argument("--dst", required=True, help="output labels dir (one <image_id>.txt per image)")
    args = parser.parse_args()
    log_params(logger, __file__, src=args.src, dst=args.dst)
    stats = convert(args.src, args.dst)
    logger.info(f"converted {stats['images']} images · {stats['polygons']} polygons · "
                f"{stats['bbox_fallback']} bbox fallbacks · {stats['crowd']} crowd skipped · "
                f"{stats['skipped']} invalid skipped -> {args.dst}")


if __name__ == "__main__":
    main()
