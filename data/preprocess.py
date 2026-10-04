"""前处理：letterbox 等比缩放 + 居中填充 + 归一化（与官方 val 口径一致）

letterbox 策略：r = min(imgsz/h, imgsz/w)，缩放尺寸取 round，填充到 imgsz 方形（灰度 114）。
"""

import cv2
import numpy as np
import torch

from config.defaults import IMGSZ

__all__ = ["letterbox", "preprocess"]


def letterbox(image, new_shape=IMGSZ, pad_color=114):
    """等比缩放 + 居中填充到方形

    Args:
        image: (H, W, 3) BGR
        new_shape: 目标边长
        pad_color: 填充灰度

    Returns:
        img: (new_shape, new_shape, 3)
        ratio: 缩放比（原图 -> 缩放后）
        pad: (pad_top, pad_left)
    """
    h, w = image.shape[:2]
    r = min(new_shape / h, new_shape / w)
    new_h, new_w = round(h * r), round(w * r)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_h, pad_w = new_shape - new_h, new_shape - new_w
    top, left = round(pad_h / 2 - 0.1), round(pad_w / 2 - 0.1)
    img = cv2.copyMakeBorder(resized, top, pad_h - top, left, pad_w - left, cv2.BORDER_CONSTANT, value=(pad_color,) * 3)
    return img, r, (top, left)


def preprocess(image_bgr, imgsz=IMGSZ):
    """BGR 图 -> 模型输入 tensor

    Returns:
        tensor: (1, 3, imgsz, imgsz) float32 [0, 1] RGB
        ratio, pad: 坐标还原参数
    """
    img, ratio, pad = letterbox(image_bgr, imgsz)
    img = np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1))[None].astype(np.float32) / 255.0  # BGR->RGB
    return torch.from_numpy(img), ratio, pad
