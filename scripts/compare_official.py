"""官方模型对照评估（dev-time）：用本工程管线跑官方权重，产出同口径完整指标

用途：README 基准表中 Official 行的数字——官方权重 + 本工程预处理/后处理/指标代码 +
本机硬件，与 Flash-YOLO 行完全同口径（唯一区别是执行代码）。

依赖：ultralytics（仅 dev-time，requirements-dev.txt）+ COCO val2017 + 官方 .pt。

用法:
    python scripts/compare_official.py --data /path/to/coco              # E2E 路径
    python scripts/compare_official.py --data /path/to/coco --nms        # NMS 路径
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # 仓库根目录入 sys.path

import numpy as np
import torch

from config.datasets import load_dataset
from config.inference import CONF_THRES, IOU_THRES, MAX_DET
from data.build import build_eval_dataset
from data.preprocess import preprocess
from eval.coco_evaluator import CocoEvaluator
from utils.engine import Detections
from utils.postprocess import non_max_suppression, scale_boxes
from utils.progress import ProgressBar


def main():
    parser = argparse.ArgumentParser(description="run the official model through Flash-YOLO's eval pipeline")
    parser.add_argument("--weights", default="weights/yolo26n.pt", help="official .pt path")
    parser.add_argument("--data", required=True,
                        help="dataset descriptor: a name in config/datasets/ (local/ wins) or a .yaml path")
    parser.add_argument("--split", default="val", help="role to evaluate (a key in the descriptor)")
    parser.add_argument("--nms", action="store_true", help="o2m+NMS path (default E2E)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int, default=0, help="evaluate first N images only (0=all)")
    args = parser.parse_args()
    print(f"scripts/compare_official.py: weights={args.weights}, data={args.data}, split={args.split}, "
          f"nms={args.nms}, limit={args.limit}")

    ultralytics = __import__("ultralytics")
    model = ultralytics.YOLO(args.weights).model.eval().to(args.device)
    head = model.model[-1]
    end2end = not args.nms
    head.end2end = end2end

    try:
        spec = load_dataset(args.data)
    except (ValueError, FileNotFoundError) as e:
        parser.error(str(e))
    dataset = build_eval_dataset(spec, args.split)
    n_total = len(dataset) if not args.limit else min(args.limit, len(dataset))
    evaluator = CocoEvaluator(dataset.gt_source(), nc=len(dataset.names))

    bar = ProgressBar(n_total, desc="official")
    for i in range(n_total):
        image = dataset.load_image(i)
        tensor, ratio, pad = preprocess(image)
        with torch.no_grad():
            out = model(tensor.to(args.device))
        out = out[0] if isinstance(out, (tuple, list)) else out
        out = out[0].cpu().numpy()
        if end2end:
            # 官方 E2E 输出 (300, 6) = [x1, y1, x2, y2, conf, cls]
            mask = out[:, 4] > CONF_THRES
            boxes, scores, cls = out[mask][:, :4], out[mask][:, 4], out[mask][:, 5].astype(np.int64)
        else:
            # 官方 o2m 输出已解码 (84, 8400)：前 4 行 xywh + 后 80 行 sigmoid 分数
            boxes, scores, cls = non_max_suppression(out[:4].T, out[4:].T, CONF_THRES, IOU_THRES, MAX_DET)
        boxes = scale_boxes(boxes, ratio, pad, *image.shape[:2])
        evaluator.update(dataset.image_id(i), Detections(boxes, scores, cls))
        if (i + 1) % 100 == 0 or i == n_total - 1:
            bar.update(i + 1)
    bar.close()

    print(f"\n官方模型 ({'E2E' if end2end else 'NMS'} 路径, {n_total} 张, Flash-YOLO 管线口径):")
    for k, v in evaluator.compute().items():
        if k in ("images", "per_class"):
            continue
        print(f"  {k:<12}: {v}")


if __name__ == "__main__":
    main()
