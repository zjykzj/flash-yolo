"""IoU 计算（numpy，xyxy 格式）"""

import numpy as np

__all__ = ["box_iou", "box_iou_matrix"]


def box_iou(box, boxes):
    """单框 vs 多框的 IoU

    Args:
        box: (4,) xyxy
        boxes: (M, 4) xyxy

    Returns:
        (M,) IoU
    """
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = (box[2] - box[0]) * (box[3] - box[1]) + area - inter
    return inter / np.maximum(union, 1e-9)


def box_iou_matrix(boxes1, boxes2):
    """两组框的 IoU 矩阵

    Args:
        boxes1: (N, 4) xyxy
        boxes2: (M, 4) xyxy

    Returns:
        (N, M) IoU
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    x1 = np.maximum(boxes1[:, None, 0], boxes2[None, :, 0])
    y1 = np.maximum(boxes1[:, None, 1], boxes2[None, :, 1])
    x2 = np.minimum(boxes1[:, None, 2], boxes2[None, :, 2])
    y2 = np.minimum(boxes1[:, None, 3], boxes2[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    union = area1[:, None] + area2[None, :] - inter
    return inter / np.maximum(union, 1e-9)
