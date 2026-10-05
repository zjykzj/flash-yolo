"""训练增强管线（numpy + cv2，最终输出 (3, imgsz, imgsz) float32 [0,1] RGB 张量）

管线顺序（与官方配方一致）：
    Mosaic(p) -> CopyPaste(p) -> RandomPerspective(必) -> MixUp(p)
    -> RandomHSV -> FlipUD(p) -> FlipLR(p) -> bgr_jitter(p) -> format

约定：
- 标签格式 (M, 5) float32 [cls, x1, y1, x2, y2]，原图像素坐标；各变换在像素空间完成
- 所有变换后标签 clip 到画布内并过滤过小框（丢弃而非填充）
- close_mosaic：trainer 在最后 N epoch 将 mosaic/mixup/copy_paste 概率原地清零
- MixUp/CopyPaste 的配对图在单样本 __getitem__ 内随机抽取（官方为 batch 内配对，
  此处按独立样本口径实现，属文档化简化）
"""

import cv2
import numpy as np
import torch

__all__ = ["augment", "letterbox_train"]

BORDER = 114  # 灰边颜色（与 letterbox 一致）


def _clip_labels(labels, w, h, min_wh=1.0):
    """clip 到画布 [0,w]x[0,h]，丢弃裁剪后过小的框"""
    if len(labels) == 0:
        return labels
    labels[:, 1] = np.clip(labels[:, 1], 0, w)
    labels[:, 2] = np.clip(labels[:, 2], 0, h)
    labels[:, 3] = np.clip(labels[:, 3], 0, w)
    labels[:, 4] = np.clip(labels[:, 4], 0, h)
    keep = (labels[:, 3] - labels[:, 1] >= min_wh) & (labels[:, 4] - labels[:, 2] >= min_wh)
    return labels[keep]


def resize_to(img, labels, size):
    """等比失真缩放到 size×size（标签同步缩放）"""
    h, w = img.shape[:2]
    out = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    lbs = labels.copy() if len(labels) else labels
    if len(lbs):
        sx, sy = size / w, size / h
        lbs[:, 1::2] *= sx
        lbs[:, 2::2] *= sy
    return out, lbs


def mosaic(img_loader, idx, n, hyp, rng):
    """4 块拼图 -> (2S, 2S) 画布

    每块图 resize 到 S×S 后按随机增益 r~U(0.5,1.5) 缩放，贴入由随机中心 (cx,cy)
    划分的象限；越界部分裁剪，标签随块变换。
    """
    S = hyp.imgsz
    canvas = np.full((S * 2, S * 2, 3), BORDER, np.uint8)
    cx = int(rng.uniform(S * 0.5, S * 1.5))
    cy = int(rng.uniform(S * 0.5, S * 1.5))
    quadrants = [(0, 0, cx, cy), (cx, 0, S * 2, cy), (0, cy, cx, S * 2), (cx, cy, S * 2, S * 2)]
    indices = [idx] + [int(rng.integers(0, n)) for _ in range(3)]

    labels4 = []
    for i, (qx0, qy0, qx1, qy1) in zip(indices, quadrants):
        img, lbs = img_loader(i)
        img, lbs = resize_to(img, lbs, S)
        r = float(rng.uniform(0.5, 1.5))
        new_size = max(int(S * r), 16)
        img = cv2.resize(img, (new_size, new_size), interpolation=cv2.INTER_LINEAR)
        if len(lbs):
            s = new_size / S
            lbs = lbs.copy()
            lbs[:, 1:] *= s
        # 贴入象限（越界裁剪）
        pw, ph = min(new_size, qx1 - qx0), min(new_size, qy1 - qy0)
        canvas[qy0 : qy0 + ph, qx0 : qx0 + pw] = img[:ph, :pw]
        if len(lbs):
            lbs[:, 1::2] += qx0
            lbs[:, 2::2] += qy0
            lbs = _clip_labels(lbs, qx1, qy1)
        labels4.append(lbs)
    return canvas, np.concatenate(labels4) if any(len(l) for l in labels4) else np.zeros((0, 5), np.float32)


def copy_paste(img, labels, img_loader, n, hyp, rng):
    """框级 CopyPaste：把另一张图的随机子集框矩形粘贴到当前画布（不做多边形光栅化）"""
    other_img, other_labels = img_loader(int(rng.integers(0, n)))
    H, W = img.shape[:2]
    other_img = cv2.resize(other_img, (W, H), interpolation=cv2.INTER_LINEAR)
    if not len(other_labels):
        return img, labels
    oh, ow = other_img.shape[:2]
    sx, sy = W / ow, H / oh
    other_labels = other_labels.copy()
    other_labels[:, 1::2] *= sx
    other_labels[:, 2::2] *= sy

    picked = other_labels[rng.random(len(other_labels)) < 0.5]
    new_labels = []
    for lb in picked:
        x1, y1, x2, y2 = lb[1:].astype(int)
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, W), min(y2, H)
        if x2 - x1 < 1 or y2 - y1 < 1:
            continue
        img[y1:y2, x1:x2] = other_img[y1:y2, x1:x2]
        new_labels.append(lb)
    if new_labels:
        labels = np.concatenate([labels, np.stack(new_labels)]) if len(labels) else np.stack(new_labels)
    return img, labels


def random_perspective(img, labels, hyp, rng):
    """随机仿射：旋转 ±degrees、缩放增益 g、剪切 ±shear、平移 ±translate*S，输出 S×S"""
    S = hyp.imgsz
    h, w = img.shape[:2]
    angle = float(rng.uniform(-hyp.degrees, hyp.degrees))
    # 缩放增益: aug_scale<1 时 g ∈ [aug_scale, 1/aug_scale]（官方口径，0.562 -> [0.562, 1.779]）
    lo, hi = (hyp.aug_scale, 1 / hyp.aug_scale) if hyp.aug_scale < 1 else (1 / hyp.aug_scale, hyp.aug_scale)
    g = float(rng.uniform(lo, hi))
    shear = float(rng.uniform(-hyp.shear, hyp.shear))
    tx = float(rng.uniform(-hyp.translate, hyp.translate)) * w
    ty = float(rng.uniform(-hyp.translate, hyp.translate)) * h

    # 旋转+缩放绕画布中心（尺寸统一到 S×S 画布）
    R = cv2.getRotationMatrix2D((w / 2, h / 2), angle, g)
    M = np.eye(3)
    M[:2] = R
    M[0, 1] += np.tan(np.deg2rad(shear))  # 剪切
    M[0, 2] += tx
    M[1, 2] += ty

    img = cv2.warpAffine(img, M[:2], (S, S), borderValue=(BORDER,) * 3)
    if not len(labels):
        return img, labels
    lbs = labels.copy()
    # 角点变换 -> 外接框
    corners = np.stack(
        [lbs[:, 1:3], np.stack([lbs[:, 3], lbs[:, 2]], 1), lbs[:, 3:5], np.stack([lbs[:, 1], lbs[:, 4]], 1)], 1
    )  # (M, 4, 2)
    new_c = corners @ M[:2, :2].T + M[:2, 2]
    lbs[:, 1] = new_c[:, :, 0].min(1)
    lbs[:, 2] = new_c[:, :, 1].min(1)
    lbs[:, 3] = new_c[:, :, 0].max(1)
    lbs[:, 4] = new_c[:, :, 1].max(1)
    return img, _clip_labels(lbs, S, S)


def mixup(img, labels, other_img, other_labels, alpha, rng):
    """图像混合：img*(1-r) + other*r，r ~ Beta(alpha, alpha)；标签拼接"""
    r = float(rng.beta(alpha, alpha))
    img = np.clip(img * (1 - r) + other_img * r, 0, 255).astype(np.uint8)
    if len(labels) and len(other_labels):
        labels = np.concatenate([labels, other_labels])
    elif len(other_labels):
        labels = other_labels
    return img, labels


def random_hsv(img, h_gain, s_gain, v_gain, rng):
    """HLS 空间通道增益（h/s/v 为增益幅度，rand(1-g, 1+g)）"""
    h, s, v = (float(rng.uniform(1 - g_, 1 + g_)) for g_ in (h_gain, s_gain, v_gain))
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    hls[..., 0] = np.clip(hls[..., 0] * h, 0, 179)
    hls[..., 1] = np.clip(hls[..., 1] * s, 0, 255)
    hls[..., 2] = np.clip(hls[..., 2] * v, 0, 255)
    return cv2.cvtColor(hls, cv2.COLOR_HLS2BGR)


def flip_lr(img, labels):
    """水平镜像：x' = W - x"""
    out = img[:, ::-1].copy()
    lbs = labels.copy()
    if len(lbs):
        W = img.shape[1]
        x1 = lbs[:, 1].copy()
        lbs[:, 1] = W - lbs[:, 3]
        lbs[:, 3] = W - x1
    return out, lbs


def flip_ud(img, labels):
    """垂直镜像：y' = H - y"""
    out = img[::-1].copy()
    lbs = labels.copy()
    if len(lbs):
        H = img.shape[0]
        y1 = lbs[:, 2].copy()
        lbs[:, 2] = H - lbs[:, 4]
        lbs[:, 4] = H - y1
    return out, lbs


def bgr_jitter(img, p, rng):
    """BGR 通道增益 + 偏置抖动（幅度 p）"""
    gains = rng.uniform(1 - p, 1 + p, 3).astype(np.float32)
    biases = rng.uniform(-p, p, 3).astype(np.float32) * 255
    out = img.astype(np.float32) * gains + biases
    return np.clip(out, 0, 255).astype(np.uint8)


def format_img(img_bgr):
    """BGR uint8 -> RGB float32 [0,1] (3,H,W) 张量"""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    t = np.ascontiguousarray(rgb.transpose(2, 0, 1)) / 255.0
    return torch.from_numpy(t.astype(np.float32))


def augment(img, labels, img_loader, hyp, rng, idx, n):
    """完整训练增强 -> (tensor (3,S,S), labels (M,5))"""
    S = hyp.imgsz
    if rng.random() < hyp.mosaic:
        img, labels = mosaic(img_loader, idx, n, hyp, rng)
    else:
        img, labels = resize_to(img, labels, S)
    if rng.random() < hyp.copy_paste and len(labels):
        img, labels = copy_paste(img, labels, img_loader, n, hyp, rng)
    img, labels = random_perspective(img, labels, hyp, rng)

    if rng.random() < hyp.mixup:
        other_img, other_labels = img_loader(int(rng.integers(0, n)))
        other_img, other_labels = resize_to(other_img, other_labels, S)
        other_img, other_labels = random_perspective(other_img, other_labels, hyp, rng)
        img, labels = mixup(img, labels, other_img, other_labels, alpha=32.0, rng=rng)

    img = random_hsv(img, hyp.hsv_h, hyp.hsv_s, hyp.hsv_v, rng)
    if rng.random() < hyp.flipud:
        img, labels = flip_ud(img, labels)
    if rng.random() < hyp.fliplr:
        img, labels = flip_lr(img, labels)
    if rng.random() < hyp.bgr:
        img = bgr_jitter(img, hyp.bgr, rng)
    return format_img(img), labels


def letterbox_train(img, labels, imgsz):
    """augment=False 路径：等比缩放 + 灰边（复用 data/preprocess.letterbox 口径）"""
    from data.preprocess import letterbox

    img, ratio, (pad_top, pad_left) = letterbox(img, new_shape=imgsz)
    lbs = labels.copy()
    if len(lbs):
        lbs[:, 1::2] = lbs[:, 1::2] * ratio + pad_left
        lbs[:, 2::2] = lbs[:, 2::2] * ratio + pad_top
        lbs = _clip_labels(lbs, imgsz, imgsz)
    return format_img(img), lbs
