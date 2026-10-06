"""训练中验证：EMA 模型快速评估（FastMetrics，COCO 口径近似，无 pycocotools 依赖）

趋势指标与 best 选取用；正式数字以 scripts/eval.py（pycocotools）为准。

控制台节奏（与训练行同机制）：调用方先打印 VAL_HEADER，进度条 desc 用实时 all 行
（图数/实例数真实增长，指标列 '-' 占位），结束时定格为真实指标行并保留。
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

__all__ = ["validate", "VAL_HEADER", "format_val_row"]

VAL_HEADER = "      " + "%11s" * 7 % ("Class", "Images", "Instances", "P", "R", "mAP50", "mAP50-95")


def format_val_row(metrics=None, images=0, instances=0):
    """val 数据行（与表头同构 11 宽右对齐）——进度条实时行与定格行共用

    metrics=None：验证进行中，指标列 '-' 占位（images/instances 传实时累计数）；
    metrics 传入时用其 images/instances 与指标填真实值。
    """
    if metrics is not None:
        images, instances = metrics["images"], metrics["instances"]
        tail = "%11.4f" * 4 % (metrics["P"], metrics["R"], metrics["mAP@50"], metrics["mAP@[.5:.95]"])
    else:
        tail = "".join(f"{'-':>11}" for _ in range(4))
    return "      " + "%11s" % "all" + "%11d" * 2 % (images, instances) + tail


def validate(model, dataset, device, conf=CONF_THRES, limit=0, batch=16):
    """EMA 模型（已 eval）在数据集上评估

    前向按 batch 合并（逐图 batch-1 前向是验证侧的主要开销），后处理/指标逐图不变。

    Returns:
        metrics dict：mAP@50 / mAP@[.5:.95] / AR@100 / P / R / images / instances
    """
    model.model[-1].end2end = True
    metrics = FastMetrics(nc=len(dataset.names))

    n = len(dataset) if not limit else min(limit, len(dataset))
    if n == 0:
        return metrics.compute()

    bar = ProgressBar(n, desc=format_val_row())
    t0 = time.monotonic()
    n_inst_seen = 0
    speed = None
    pending = []  # (idx, image, tensor, ratio, pad)
    for i in range(n):
        image = dataset.load_image(i)
        img, ratio, pad = preprocess(image)
        pending.append((i, image, img, ratio, pad))
        if len(pending) < batch and i != n - 1:
            continue
        with torch.no_grad():
            outs = model(torch.cat([p[2] for p in pending], 0).to(device)).cpu().numpy()  # (b, 300, 6)
        for (j, img_j, _t, ratio, pad), out in zip(pending, outs):
            mask = out[:, 4] > conf
            boxes = scale_boxes(out[mask][:, :4], ratio, pad, *img_j.shape[:2])
            dets = Detections(boxes, out[mask][:, 4], out[mask][:, 5].astype("int64"))
            gt = np.array([[x1, y1, x2, y2, c] for (x1, y1, x2, y2), c in dataset.targets(j, include_crowd=False)], np.float32)
            gt = gt.reshape(-1, 5)
            n_inst_seen += len(gt)
            metrics.update(
                dets.boxes, dets.scores, dets.class_ids,
                gt[:, :4] if len(gt) else np.zeros((0, 4), np.float32),
                gt[:, 4].astype(np.int64) if len(gt) else np.zeros(0, np.int64),
            )
        pending.clear()
        if (i + 1) % 10 == 0 or i == n - 1:
            speed = (i + 1) / max(time.monotonic() - t0, 1e-6)
            bar.update(i + 1, speed, desc=format_val_row(images=i + 1, instances=n_inst_seen))
    result = metrics.compute()
    # 定格：末次刷新换成真实指标行（保留末段速度），换行保留（与训练行同机制）
    bar.update(n, speed, desc=format_val_row(result))
    bar.close()
    return result
