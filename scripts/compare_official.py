"""官方模型对照评估（dev-time）：用本工程管线跑官方权重，产出同口径完整指标

用途：README 基准表中 Official 行的数字——官方权重 + 本工程预处理/后处理/指标代码 +
本机硬件，与 Flash-YOLO 行完全同口径（唯一区别是执行代码）。分割权重同时输出 Box/Mask
两张表（双输出：检测+系数、原型）。

依赖：ultralytics（仅 dev-time，requirements-dev.txt）+ COCO val2017 + 官方 .pt。

用法:
    python scripts/compare_official.py --data /path/to/coco              # E2E 路径
    python scripts/compare_official.py --data /path/to/coco --nms        # NMS 路径
    python scripts/compare_official.py --weights yolo26n-seg.pt --data /path/to/coco  # 分割
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
from utils.engines import Detections
from utils.postprocess import non_max_suppression, process_mask, scale_boxes
from utils.progress import ProgressBar

_HEADER = ("      " + f"{'Class':<15}" + "".join(
    f"{c:>{w}}" for c, w in zip(("Images", "Instances", "P", "R", "mAP50", "mAP50-95", "AR@100"),
                                (7, 10, 10, 10, 10, 10, 10))))


def _fmt_row(name, n_img, n_inst, p, r, ap50, ap, ar):
    return ("      " + f"{str(name)[:15]:<15}" + f"{n_img:>7d}{n_inst:>10d}"
            + "".join(f"{v:>10.4f}" for v in (p, r, ap50, ap, ar)))


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
    task = getattr(model, "task", "detect")

    try:
        spec = load_dataset(args.data)
    except (ValueError, FileNotFoundError) as e:
        parser.error(str(e))
    dataset = build_eval_dataset(spec, args.split)
    n_total = len(dataset) if not args.limit else min(args.limit, len(dataset))
    heads = ["Box", "Mask"] if task == "segment" else ["Box"]
    evaluators = [(name, CocoEvaluator(dataset.gt_source(), nc=len(dataset.names),
                                       iou_type="segm" if name == "Mask" else "bbox"))
                  for name in heads]

    bar = ProgressBar(n_total, desc="official")
    for i in range(n_total):
        image = dataset.load_image(i)
        tensor, ratio, pad = preprocess(image)
        with torch.no_grad():
            out = model(tensor.to(args.device))
        proto = coeff = None
        # 官方 8.4.173 输出统一包一层：((检测输出[, 原型]), {one2many, one2one})
        out = out[0] if isinstance(out, (tuple, list)) else out
        if task == "segment":
            # 分割：out = (检测输出, 原型)；检测部分与检测模型同口径
            preds, proto = out
            preds = preds[0].cpu().numpy()
            proto = proto[0].cpu().numpy()
        else:
            preds = out[0].cpu().numpy()
        if end2end:
            # 官方 E2E 输出 (300, 6 [+ nm]) = [x1, y1, x2, y2, conf, cls, 系数...]
            keep = preds[:, 4] > CONF_THRES
            boxes, scores, cls = preds[keep][:, :4], preds[keep][:, 4], preds[keep][:, 5].astype(np.int64)
            if task == "segment":
                coeff = preds[keep][:, 6:]
        elif task == "segment":
            # 官方 o2m 输出 (4+nc+nm, N)：前 4 行 xywh，随后 nc 行分数，末尾 nm 行系数
            nc = preds.shape[0] - 4 - proto.shape[0]
            boxes, scores, cls, coeff = non_max_suppression(
                preds[:4].T, preds[4:4 + nc].T, CONF_THRES, IOU_THRES, MAX_DET, extra=preds[4 + nc:].T)
        else:
            # 官方 o2m 输出已解码 (84, 8400)：前 4 行 xywh + 后 80 行 sigmoid 分数
            boxes, scores, cls = non_max_suppression(preds[:4].T, preds[4:].T, CONF_THRES, IOU_THRES, MAX_DET)
        masks = None
        if task == "segment":
            h, w = tensor.shape[-2:]
            masks = process_mask(proto, coeff, boxes, (h, w), image.shape[:2], ratio, pad)
        boxes = scale_boxes(boxes, ratio, pad, *image.shape[:2])
        dets = Detections(boxes, scores, cls, masks)
        for _name, evaluator in evaluators:
            evaluator.update(dataset.image_id(i), dets)
        if (i + 1) % 100 == 0 or i == n_total - 1:
            bar.update(i + 1)
    bar.close()

    print(f"\n官方模型 ({'E2E' if end2end else 'NMS'} 路径, {n_total} 张, Flash-YOLO 管线口径):")
    for name, evaluator in evaluators:
        metrics = evaluator.compute()
        if len(evaluators) > 1:
            print(f"\n  {name} metrics")
        print(_HEADER)
        n_inst = sum(r[2] for r in metrics["per_class"])
        print(_fmt_row("all", metrics["images"], n_inst, metrics["P"], metrics["R"],
                       metrics["mAP@50"], metrics["mAP@[.5:.95]"], metrics["AR@100"]))
        for row in metrics["per_class"]:
            print(_fmt_row(*row))


if __name__ == "__main__":
    main()
