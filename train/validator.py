"""训练中验证：EMA 模型快速评估（FastMetrics，COCO 口径近似，无 pycocotools 依赖）

趋势指标与 best 选取用；正式数字以 scripts/eval.py（pycocotools）为准。
"""

import time

import numpy as np
import torch

from config.defaults import CONF_THRES
from data.preprocess import preprocess
from train.metrics import FastMetrics
from utils.engine import Detections
from utils.postprocess import scale_boxes
from utils.progress import ProgressBar

__all__ = ["validate"]


def validate(model, dataset, device, conf=CONF_THRES, limit=0):
    """EMA 模型（已 eval）在数据集上评估

    Returns:
        metrics dict：mAP@50 / mAP@[.5:.95] / AR@100 / images / instances
    """
    model.model[-1].end2end = True
    metrics = FastMetrics(nc=len(dataset.names))

    n = len(dataset) if not limit else min(limit, len(dataset))
    if n == 0:
        return metrics.compute()

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
        gt = np.array([[x1, y1, x2, y2, c] for (x1, y1, x2, y2), c in dataset.targets(i, include_crowd=False)], np.float32)
        gt = gt.reshape(-1, 5)
        metrics.update(
            dets.boxes, dets.scores, dets.class_ids,
            gt[:, :4] if len(gt) else np.zeros((0, 4), np.float32),
            gt[:, 4].astype(np.int64) if len(gt) else np.zeros(0, np.int64),
        )
        if (i + 1) % 10 == 0 or i == n - 1:
            bar.update(i + 1, (i + 1) / max(time.monotonic() - t0, 1e-6), desc=f"val {i + 1}/{n}")
    bar.close(clear=True)  # 进度条清屏，val 表行（调用方 logger）是唯一记录
    return metrics.compute()
