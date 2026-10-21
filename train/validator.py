"""训练中验证：EMA 模型快速评估（FastMetrics，COCO 口径近似，无 pycocotools 依赖）

趋势指标与 best 选取用；正式数字以 scripts/eval.py（pycocotools）为准。

控制台节奏（与训练行同机制）：调用方先打印 VAL_HEADER / VAL_HEADER_SEG，进度条 desc 用实时
all 行（图数/实例数真实增长，指标列 '-' 占位），结束时定格为真实指标行并保留。
val 损失（可选，传入 loss_fn 时启用）：与训练完全相同的损失实现（ComputeLoss /
ComputeLossSeg / ComputeLossV3），复用同一次 backbone/neck 前向（forward_feats → head 训练
口径输出 → 损失与指标双用途，eval 模式 BN 不污染统计）；逐 batch 累加、按 batch 数平均，
返回的 result["loss"] 键随损失实现（item_keys），**仅写入 results.csv，控制台不展示**。

指标侧的解码/后处理由 head.postprocess_val(preds, feats) 统一提供（每图 (M,6)[+nm]
[xyxy, conf, cls, 系数...]，letterbox 像素）：Detect 走 E2E o2o 解码，Segment26 额外用
低分辨率 process_mask_lowres 还原掩码，V3Detect 解码 + 按类 NMS。
"""

import time

import numpy as np
import torch

from config.inference import CONF_THRES, IMGSZ
from data.preprocess import preprocess
from train.metrics import FastMetrics
from utils.engines import Detections
from utils.postprocess import process_mask_lowres, scale_boxes
from utils.progress import ProgressBar

__all__ = ["validate", "VAL_HEADER", "VAL_HEADER_SEG", "format_val_row"]

VAL_HEADER = "      " + "%11s" * 7 % ("Class", "Images", "Instances", "P", "R", "mAP50", "mAP50-95")
VAL_HEADER_SEG = "      " + "%11s" * 11 % (
    "Class", "Images", "Instances", "P", "R", "mAP50", "mAP50-95",
    "P(M)", "R(M)", "mAP50(M)", "mAP50-95(M)")

_LOSS_KEYS = ("box", "cls", "l1", "o2m", "o2o")  # 无 loss_fn 时的兜底键（正常路径由 item_keys 派生）
_LOSS_KEYS_SEG = ("box", "seg", "cls", "l1", "o2m", "o2o")


def format_val_row(metrics=None, images=0, instances=0, seg=False):
    """val 数据行（与表头同构 11 宽右对齐）——进度条实时行与定格行共用"""
    if metrics is not None:
        images, instances = metrics["images"], metrics["instances"]
        tail = "%11.4f" * 4 % (metrics["P"], metrics["R"], metrics["mAP@50"], metrics["mAP@[.5:.95]"])
        if seg:
            tail += "%11.4f" * 4 % (metrics["P(M)"], metrics["R(M)"],
                                    metrics["mAP@50(M)"], metrics["mAP@[.5:.95](M)"])
    else:
        tail = "".join(f"{'-':>11}" for _ in range(8 if seg else 4))
    return "      " + "%11s" % "all" + "%11d" * 2 % (images, instances) + tail


def _gt_boxes_letterbox_to_original(gt_lb, ratio, pad, h, w):
    """seg_val 的 letterbox 像素框 -> 原图像素框（与 scale_boxes 互逆）"""
    boxes = gt_lb[:, 1:5].copy()
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad[1]) / ratio
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad[0]) / ratio
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h)
    return boxes


def validate(model, dataset, device, conf=CONF_THRES, limit=0, batch=16, loss_fn=None, imgsz=IMGSZ,
             task="detect", mask_ratio=4, overlap_mask=True, return_metrics=False):
    """EMA 模型（已 eval）在数据集上评估

    前向按 batch 合并（逐图 batch-1 前向是验证侧的主要开销），后处理/指标逐图不变；
    loss_fn 传入时同一批特征（forward_feats + head 训练口径双分支输出）另算 val 损失。
    task="segment" 时额外返回 Mask 指标（训练用 imgsz/mask_ratio 网格的快速近似）。

    Returns:
        metrics dict：mAP@50 / mAP@[.5:.95] / AR@100 / P / R / images / instances
        （segment 另有 P/R/mAP@50/mAP@[.5:.95](M)；loss_fn 非空时另有 result["loss"]）
    """
    head = model.model[-1]
    if hasattr(head, "end2end"):
        head.end2end = True  # YOLO26 走 E2E 解码；V3Detect 无此开关
    is_seg = task == "segment"
    metrics = FastMetrics(nc=len(dataset.names), task=task)
    loss_keys = tuple(loss_fn.item_keys) if loss_fn is not None else (
        _LOSS_KEYS_SEG if is_seg else _LOSS_KEYS)

    n = len(dataset) if not limit else min(limit, len(dataset))
    if n == 0:
        result = metrics.compute()
        if loss_fn is not None:
            result["loss"] = dict.fromkeys(loss_keys, 0.0)
        return (result, metrics) if return_metrics else result

    bar = ProgressBar(n, desc=format_val_row(seg=is_seg))
    t0 = time.monotonic()
    n_inst_seen = 0
    speed = None
    loss_sums = dict.fromkeys(loss_keys, 0.0)
    n_batches = 0
    # (idx, image, tensor, ratio, pad, seg_extra)；seg_extra = (gt_lb_loss, mask_loss,
    # gt_lb_metric, mask_metric)，检测为 None
    pending = []
    for i in range(n):
        image = dataset.load_image(i)
        img, ratio, pad = preprocess(image, imgsz)  # letterbox 尺寸 = 训练 imgsz（与损失锚点一致）
        seg_extra = None
        if is_seg:
            # 一次多边形解析 + 一次逐实例 fillPoly：loss 网格与指标网格共用（此前调两次
            # seg_val，默认 mask_ratio=4 时两次栅格化完全相同）
            (gt_lb_loss, mask_loss), (gt_lb_metric, mask_metric) = dataset.seg_val_multi(
                i, imgsz, ratio, pad, [(mask_ratio, overlap_mask), (4, False)])
            seg_extra = (gt_lb_loss, mask_loss, gt_lb_metric, mask_metric)
        pending.append((i, image, img, ratio, pad, seg_extra))
        if len(pending) < batch and i != n - 1:
            continue
        with torch.no_grad():
            feats = model.forward_feats(torch.cat([p[2] for p in pending], 0).to(device))
            preds = head._forward_train(feats)  # 训练口径原始输出（eval 模式 BN，更新统计已停用）
            outs = head.postprocess_val(preds, feats)  # 每图 (M, 6[+nm])（letterbox 像素）
            if loss_fn is not None:
                # 目标变换到 letterbox 空间（scale_boxes 的逆变换），与训练损失同像素口径
                tgt, seg_masks = [], []
                for bi, (j, _img, _t, ratio, pad, extra) in enumerate(pending):
                    if is_seg:
                        gt_lb = extra[0]
                        for c, x1, y1, x2, y2 in gt_lb:
                            tgt.append([bi, c, x1, y1, x2, y2])
                        seg_masks.append(extra[1])
                    else:
                        for (x1, y1, x2, y2), c in dataset.targets(j, include_crowd=False):
                            tgt.append([bi, c, x1 * ratio + pad[1], y1 * ratio + pad[0],
                                        x2 * ratio + pad[1], y2 * ratio + pad[0]])
                tg = (torch.tensor(tgt, dtype=torch.float32, device=device) if tgt
                      else torch.zeros((0, 6), device=device))
                if is_seg:
                    masks = torch.from_numpy(np.concatenate(seg_masks, 0)).to(device)
                    _, items = loss_fn(preds, tg, len(pending), imgsz, masks=masks)
                else:
                    _, items = loss_fn(preds, tg, len(pending), imgsz)
                for k in loss_sums:
                    loss_sums[k] += items[k]
                n_batches += 1
        proto = preds["one2one"]["proto"] if is_seg else None
        for bi, (j, img_j, _t, ratio, pad, extra) in enumerate(pending):
            out = outs[bi]
            keep = out[:, 4] > conf
            boxes_lb = out[keep][:, :4]
            scores = out[keep][:, 4]
            cls = out[keep][:, 5].astype(np.int64)
            boxes = scale_boxes(boxes_lb, ratio, pad, *img_j.shape[:2])
            dets = Detections(boxes, scores, cls)
            if is_seg:
                coeff = out[keep][:, 6:]
                det_masks = process_mask_lowres(
                    proto[bi].detach().float().cpu().numpy(), coeff, boxes_lb, (imgsz, imgsz))
                gt_lb_m, gt_masks = extra[2], extra[3]
                gt_cls = gt_lb_m[:, 0].astype(np.int64)
                gt_boxes = _gt_boxes_letterbox_to_original(gt_lb_m, ratio, pad, *img_j.shape[:2])
                metrics.update(dets.boxes, dets.scores, dets.class_ids, gt_boxes, gt_cls)
                metrics.update_mask(det_masks, dets.scores, dets.class_ids, gt_masks, gt_cls)
                n_inst_seen += len(gt_cls)
            else:
                gt = np.array([[x1, y1, x2, y2, c] for (x1, y1, x2, y2), c in
                               dataset.targets(j, include_crowd=False)], np.float32)
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
            bar.update(i + 1, speed, desc=format_val_row(images=i + 1, instances=n_inst_seen, seg=is_seg))
    result = metrics.compute()
    # 定格：末次刷新换成真实指标行（保留末段速度），换行保留（与训练行同机制）
    bar.update(n, speed, desc=format_val_row(result, seg=is_seg))
    bar.close()
    if loss_fn is not None:
        result["loss"] = {k: v / max(n_batches, 1) for k, v in loss_sums.items()}
    return (result, metrics) if return_metrics else result
