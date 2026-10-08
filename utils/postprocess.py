"""后处理（NMS 路径）：解码 + 按类 NMS（numpy 实现，pt/onnx 引擎共用）

E2E 路径的 top-k 解码已在模型图内完成（model/head.py），此处只负责：
    - NMS 路径：raw (4+nc, N) -> 解码 -> 按类 NMS
    - 坐标还原到原图（两条路径共用）
"""

import numpy as np

from config.inference import CONF_THRES, IOU_THRES, MAX_DET
from utils.iou import box_iou

__all__ = ["decode_raw", "non_max_suppression", "nms_per_image", "v3_detections", "scale_boxes"]

# 三个检测层特征图尺寸（640 输入）：P3/P4/P5
_LEVEL_SHAPES = [(80, 80), (40, 40), (20, 20)]
_STRIDES = (8, 16, 32)


def make_anchors_np(shapes, strides, offset=0.5):
    """anchor 点（grid 单位），(2, N)/(1, N) 转置形式（与 utils/anchors.py 的 torch 版一致）"""
    anchors, stride_t = [], []
    for (h, w), s in zip(shapes, strides):
        sx = np.arange(w, dtype=np.float32) + offset
        sy = np.arange(h, dtype=np.float32) + offset
        sy, sx = np.meshgrid(sy, sx, indexing="ij")
        anchors.append(np.stack((sx, sy), -1).reshape(-1, 2))
        stride_t.append(np.full((h * w, 1), s, dtype=np.float32))
    return np.concatenate(anchors).T, np.concatenate(stride_t).T


def dist2bbox_np(distance, anchors, xywh=True):
    """ltrb (4, N) -> 框 (4, N)（grid 单位，未乘 stride）

    默认 xywh（与官方 o2m 解码格式一致），xywh=False 输出 xyxy。
    """
    lt, rb = np.split(distance, 2, axis=0)
    x1y1 = anchors - lt
    x2y2 = anchors + rb
    if xywh:
        return np.concatenate(((x1y1 + x2y2) / 2, x2y2 - x1y1), axis=0)
    return np.concatenate((x1y1, x2y2), axis=0)


def xywh2xyxy(boxes):
    """(N, 4) xywh -> xyxy"""
    out = boxes.copy()
    out[:, 0], out[:, 1] = boxes[:, 0] - boxes[:, 2] / 2, boxes[:, 1] - boxes[:, 3] / 2
    out[:, 2], out[:, 3] = boxes[:, 0] + boxes[:, 2] / 2, boxes[:, 1] + boxes[:, 3] / 2
    return out


def decode_raw(raw, nc=80, shapes=None, strides=None):
    """解码模型原始输出

    Args:
        raw: (4+nc, N) ltrb + logits
        nc: 类别数

    Returns:
        boxes: (N, 4) xywh 像素坐标（与官方解码格式一致，NMS 前需转 xyxy）
        scores: (N, nc) sigmoid 概率
    """
    shapes = shapes or _LEVEL_SHAPES
    strides = strides or _STRIDES
    anchors, stride_t = make_anchors_np(shapes, strides)
    dbox = dist2bbox_np(raw[:4], anchors) * stride_t  # (4, N) 像素
    scores = 1.0 / (1.0 + np.exp(-raw[4 : 4 + nc]))  # sigmoid
    return dbox.T, scores.T


def _nms(boxes, scores, iou_thres):
    """单类 NMS（按分数降序贪心抑制）"""
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        order = order[1:][box_iou(boxes[i], boxes[order[1:]]) < iou_thres]
    return np.asarray(keep, dtype=np.int64)


def nms_per_image(boxes, score_map, conf_thres=CONF_THRES, iou_thres=IOU_THRES, max_det=MAX_DET):
    """按类 NMS（单图，xyxy 输入）

    Args:
        boxes: (N, 4) xyxy
        score_map: (N, nc) 概率
    Returns:
        (M, 6) [x1, y1, x2, y2, score, cls]（按分降序、≤ max_det；无检出时 (0, 6)）
    """
    nc = score_map.shape[1]
    out = []
    for c in range(nc):
        mask = score_map[:, c] > conf_thres
        b, s = boxes[mask], score_map[mask, c]
        if len(b) == 0:
            continue
        keep = _nms(b, s, iou_thres)
        out.append(np.concatenate([b[keep], s[keep, None], np.full((len(keep), 1), c, np.float32)], axis=1))
    if not out:
        return np.zeros((0, 6), dtype=np.float32)
    merged = np.concatenate(out)
    top = merged[:, 4].argsort()[::-1][:max_det]
    return merged[top]


def non_max_suppression(boxes, score_map, conf_thres=CONF_THRES, iou_thres=IOU_THRES, max_det=MAX_DET):
    """按类 NMS（与官方 val 口径：conf 0.001 / iou 0.7 / 每图最多 300）

    Args:
        boxes: (N, 4) xywh（decode_raw 输出）
        score_map: (N, nc)

    Returns:
        boxes: (M, 4) xyxy, scores: (M,), class_ids: (M,) int
    """
    det = nms_per_image(xywh2xyxy(boxes), score_map, conf_thres, iou_thres, max_det)
    return det[:, :4], det[:, 4], det[:, 5].astype(np.int64)


def v3_detections(output, conf_thres=CONF_THRES, iou_thres=IOU_THRES, max_det=MAX_DET):
    """YOLOv3-tiny 解码输出 -> NMS 后检测（单图）

    Args:
        output: (N, 5+nc) [x1,y1,x2,y2, obj, cls...]（V3Detect eval 前向 / postprocess_val 的单图行）
    Returns:
        (M, 6) [x1,y1,x2,y2, obj×cls, cls]（最终分 = obj 与类分相乘，darknet 口径）
    """
    score_map = output[:, 4:5] * output[:, 5:]
    return nms_per_image(output[:, :4], score_map, conf_thres, iou_thres, max_det)


def scale_boxes(boxes, ratio, pad, ori_h, ori_w):
    """letterbox 坐标 -> 原图坐标（并裁剪到图像内）

    Args:
        boxes: (N, 4) xyxy（letterbox 空间）
        ratio: 缩放比；pad: (top, left)
    """
    if len(boxes) == 0:
        return boxes
    boxes = boxes.copy()
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad[1]) / ratio
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad[0]) / ratio
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, ori_w)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, ori_h)
    return boxes
