"""训练中验证：EMA 模型在 COCO val 上跑 mAP（复用 M2 的 CocoEvaluator 管线）"""

import time

import torch

from config.defaults import CONF_THRES
from data.coco import CocoDataset
from data.preprocess import preprocess
from eval.coco_evaluator import CocoEvaluator
from utils.engine import Detections
from utils.postprocess import scale_boxes
from utils.progress import ProgressBar

__all__ = ["validate"]


def validate(model, dataset, device, conf=CONF_THRES, limit=0):
    """EMA 模型（已 eval）在数据集上评估

    Args:
        model: EMA 副本（eval 模式；本函数会显式设 head.end2end = True）
        dataset: CocoDataset
        device: 推理设备
        conf: 置信度阈值
        limit: 只评估前 N 张（0 = 全部）
    Returns:
        metrics dict（CocoEvaluator.compute() 口径）
    """
    model.model[-1].end2end = True
    evaluator = CocoEvaluator(dataset.ann_file)

    n = len(dataset) if not limit else min(limit, len(dataset))
    if n == 0:
        return evaluator.compute()

    bar = ProgressBar(n, desc="val")
    t0 = time.monotonic()
    for i in range(n):
        image = dataset.load_image(i)
        img, ratio, pad = preprocess(image)
        with torch.no_grad():
            out = model(img.to(device))[0].cpu().numpy()  # (300, 6)
        mask = out[:, 4] > conf
        boxes = scale_boxes(out[mask][:, :4], ratio, pad, *image.shape[:2])
        dets = Detections(boxes, out[mask][:, 4], out[mask][:, 5].astype("int64"))
        evaluator.update(dataset.image_id(i), dets)
        if (i + 1) % 10 == 0 or i == n - 1:
            bar.update(i + 1, (i + 1) / max(time.monotonic() - t0, 1e-6), desc=f"val {i + 1}/{n}")
    bar.close()
    return evaluator.compute()
