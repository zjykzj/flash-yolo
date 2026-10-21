"""训练增强管线（numpy + cv2，最终输出 (3, imgsz, imgsz) float32 [0,1] RGB 张量）

管线顺序（对齐官方 v8_transforms + Format）：
    Mosaic(p) -> CopyPaste(p) -> RandomPerspective(必) -> MixUp(p)
    -> RandomHSV -> FlipUD(p) -> FlipLR(p) -> BGR 通道互换(p) -> format

几何口径逐行对照 ultralytics/data/augment.py + data/base.py（8.4.173）：
- 载图后**等比**缩放到长边 = imgsz（官方 rect_mode；不拉伸成方形）
- Mosaic：每块按自然尺寸锚定在随机中心 (xc,yc) 贴入 2S×2S 画布、裁到画布边界
  （Mosaic._mosaic4），不做逐块随机缩放；拼接后按 _cat_labels：clip 到 2S + 丢弃零面积框
- RandomPerspective：M = T @ S @ R @ C 映射到 S×S（与输入尺寸无关，官方
  _compute_affine_matrix 同构），平移量 = U(0.5±translate)*S（**输出尺寸**）；
  仿射后按官方 box_candidates 过滤：w>2 且 h>2 且面积保留率>0.10 且长宽比<100
  （检测任务 area_thr=0.10；曾只丢 <1px 框，导致大量"只剩碎片"的框进训练）
- RandomHSV：HSV 空间 LUT（hue 加性偏移 (x+r*180)%180、sat/val 乘性、lut_sat[0]=0）；
  曾在 HLS 空间做且把 s/v 增益错位到 L/V 通道
- bgr：以概率 p 把最终张量的通道序换成 BGR（官方 Format：源图 BGR，bgr 表示"返回 BGR 的概率"，
  默认 0 = 恒为 RGB）
- copy_paste：官方 CopyPaste 要求 instances.segments 非空，纯检测标签下**恒为 no-op**；
  本仓库默认 copy_paste_mode="off" 对齐该口径（"box" 为矩形贴块近似，会引入标签噪声）

约定：
- 标签格式 (M, 5) float32 [cls, x1, y1, x2, y2]，当前画布像素坐标
- 分割：segments list[(K,2)] 与标签行一一对应，几何算子同步变换；最后一个算子后
  在 S×S 栅格化并按 mask_ratio 降采样（overlap=True 时按面积降序重排标签）
- close_mosaic：trainer 在最后 N epoch 将 mosaic/mixup/copy_paste 概率原地清零
- MixUp/CopyPaste 的配对图在单样本 __getitem__ 内随机抽取（官方为 batch 内配对，
  此处按独立样本口径实现，属文档化简化）
"""

import cv2
import numpy as np
import torch

from data.segments import polygon_bbox, rasterize_segments

__all__ = ["augment", "letterbox_train", "resize_long_side"]

BORDER = 114  # 灰边颜色（与 letterbox 一致）
WH_THR = 2.0  # box_candidates: 最小宽/高（px，官方默认）
AR_THR = 100.0  # box_candidates: 最大长宽比
AREA_THR = 0.10  # box_candidates: 最小面积保留率（检测任务，官方口径）


# ---- 几何工具（官方口径）----
def _resize_long_side(img, labels, segments, size):
    h0, w0 = img.shape[:2]
    if max(h0, w0) == size:
        return img, labels, segments
    r = size / max(h0, w0)
    w, h = min(int(np.ceil(w0 * r)), size), min(int(np.ceil(h0 * r)), size)
    out = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    segs = _scale_segments(segments, w / w0, h / h0) if segments is not None else None
    if not len(labels):
        return out, labels, segs
    lbs = labels.copy()
    lbs[:, 1::2] *= w / w0
    lbs[:, 2::2] *= h / h0
    return out, lbs, segs


def resize_long_side(img, labels, size, segments=None):
    """等比缩放到长边 = size（官方 BaseDataset.load_image rect_mode）

    小图同样上采样；尺寸取 ceil 并 clamp 到 size（官方 min(math.ceil(w0*r), imgsz)）。
    segments 给定时返回三元组（多边形按同一比例缩放）。
    """
    out, lbs, segs = _resize_long_side(img, labels, segments, size)
    return _ret(out, lbs, segs)


def _clip_boxes(boxes, w, h):
    """clip (M,4) xyxy 到 [0,w]×[0,h]（官方 Instances.clip）"""
    if not len(boxes):
        return boxes
    boxes[:, 0] = np.clip(boxes[:, 0], 0, w)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, w)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, h)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, h)
    return boxes


def _clip(labels, w, h):
    """clip (M,5) [cls, xyxy] 标签"""
    _clip_boxes(labels[:, 1:5], w, h)
    return labels


def _zero_area_keep(labels):
    """零面积框的保留掩码（标签与多边形必须共用同一个 keep）"""
    if not len(labels):
        return np.zeros(0, bool)
    return (labels[:, 3] - labels[:, 1]) * (labels[:, 4] - labels[:, 2]) > 0


def _drop_zero_area(labels):
    """丢弃零面积框（官方 Instances.remove_zero_area_boxes）"""
    if not len(labels):
        return labels
    return labels[_zero_area_keep(labels)]


# ---- 多边形几何（与框逐点对齐；segments=None 时全部跳过）----
def _scale_segments(segments, sx, sy):
    return [np.stack([p[:, 0] * sx, p[:, 1] * sy], 1).astype(np.float32) for p in segments]


def _shift_segments(segments, dx, dy):
    return [(p + np.array([dx, dy], np.float32)).astype(np.float32) for p in segments]


def _clip_segments(segments, w, h):
    return [np.stack([np.clip(p[:, 0], 0, w), np.clip(p[:, 1], 0, h)], 1).astype(np.float32)
            for p in segments]


def _affine_segments(segments, M, perspective=False):
    """多边形逐点变换；perspective=True 时按齐次坐标除以第三维（官方 apply_segments 口径）"""
    out = []
    for p in segments:
        hom = np.concatenate([p, np.ones((len(p), 1), np.float32)], 1)
        q = hom @ M.T
        if perspective:
            q = q[:, :2] / q[:, 2:3]
        else:
            q = q[:, :2]
        out.append(q.astype(np.float32))
    return out


def _transform_boxes(corners, M, perspective=False):
    """(M, 4, 2) 角点 -> 变换后外接框；perspective 时齐次归一化（官方 apply_bboxes 口径）"""
    hom = np.concatenate([corners, np.ones(corners.shape[:2] + (1,), np.float32)], 2)
    q = hom @ M.T
    if perspective:
        q = q[..., :2] / q[..., 2:3]
    else:
        q = q[..., :2]
    return np.stack([q[:, :, 0].min(1), q[:, :, 1].min(1),
                     q[:, :, 0].max(1), q[:, :, 1].max(1)], 1).astype(np.float32)


def _flip_lr_segments(segments, W):
    return [np.stack([W - p[:, 0], p[:, 1]], 1).astype(np.float32) for p in segments]


def _flip_ud_segments(segments, H):
    return [np.stack([p[:, 0], H - p[:, 1]], 1).astype(np.float32) for p in segments]


def _subset_segments(segments, keep):
    return [p for p, k in zip(segments, keep) if k]


def _ret(img, labels, segments):
    """内部 3 元组 -> 公共 API：segments 未启用时维持历史 2 元组"""
    return (img, labels, segments) if segments is not None else (img, labels)


def _load_sample(img_loader, index, with_segments):
    """邻图加载：分割数据集返回 3 元组，检测数据集返回 2 元组"""
    sample = img_loader(index)
    if with_segments:
        return sample
    img, labels = sample
    return img, labels, None


def _box_candidates(boxes_pre, boxes_post, wh_thr=WH_THR, ar_thr=AR_THR, area_thr=AREA_THR, eps=1e-16):
    """官方 RandomPerspective.box_candidates：仿射后保留仍"实质可见"的框

    Args:
        boxes_pre: (M,4) 变换前框 × 仿射 scale（官方 box1 口径，用于面积保留率）
        boxes_post: (M,4) 变换 + clip 后的框
    """
    if not len(boxes_post):
        return np.zeros(0, bool)
    w1 = np.maximum(boxes_pre[:, 2] - boxes_pre[:, 0], 0)
    h1 = np.maximum(boxes_pre[:, 3] - boxes_pre[:, 1], 0)
    w2 = boxes_post[:, 2] - boxes_post[:, 0]
    h2 = boxes_post[:, 3] - boxes_post[:, 1]
    ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))
    return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + eps) > area_thr) & (ar < ar_thr)


# ---- 增强算子 ----
def _mosaic(img_loader, idx, n, hyp, rng, segments):
    S = hyp.imgsz
    canvas = np.full((S * 2, S * 2, 3), BORDER, np.uint8)
    b = -(S // 2)  # 官方 self.border = (-imgsz // 2, -imgsz // 2)
    yc, xc = (int(rng.uniform(-b, 2 * S + b)) for _ in range(2))
    indices = [idx] + [int(rng.integers(0, n)) for _ in range(3)]

    labels4, segs4 = [], []
    for i, index in enumerate(indices):
        img, lbs, segs = _load_sample(img_loader, index, segments is not None)
        img, lbs, segs = _resize_long_side(img, lbs, segs, S)
        h, w = img.shape[:2]
        if i == 0:  # top left
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
        elif i == 1:  # top right
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, S * 2), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # bottom left
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(S * 2, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
        else:  # bottom right
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, S * 2), min(S * 2, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
        canvas[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
        dx, dy = x1a - x1b, y1a - y1b
        if len(lbs):
            lbs = lbs.copy()
            lbs[:, 1::2] += dx  # 官方 _update_labels(padw, padh)
            lbs[:, 2::2] += dy
        labels4.append(lbs)
        if segs is not None:
            segs4.append(_shift_segments(segs, dx, dy))
    labels = np.concatenate(labels4) if any(len(l) for l in labels4) else np.zeros((0, 5), np.float32)
    if segments is None:
        return canvas, _drop_zero_area(_clip(labels, S * 2, S * 2)), None
    labels = _clip(labels, S * 2, S * 2)
    keep = _zero_area_keep(labels)
    segs = _subset_segments([p for part in segs4 for p in part], keep)
    return canvas, labels[keep], _clip_segments(segs, S * 2, S * 2)


def mosaic(img_loader, idx, n, hyp, rng, segments=None):
    """4 块拼图 -> (2S, 2S) 画布（官方 Mosaic._mosaic4 几何）

    每块等比缩放到长边 S 后按自然尺寸锚定在随机中心 (xc, yc)：左/上块以中心为右下角、
    右/下块以中心为左上角，越界部分裁到画布（块可以跨越中线）。
    segments 给定时返回三元组，多边形与框共享同一 keep。
    """
    return _ret(*_mosaic(img_loader, idx, n, hyp, rng, segments))


def _bbox_ioa(boxes1, boxes2, eps=1e-7):
    """intersection / area(boxes2)（官方 bbox_ioa 口径；boxes1 = 待粘贴实例）"""
    if not len(boxes1) or not len(boxes2):
        return np.zeros((len(boxes1), len(boxes2)), np.float32)
    inter = (np.minimum(boxes1[:, None, 2], boxes2[None, :, 2])
             - np.maximum(boxes1[:, None, 0], boxes2[None, :, 0])).clip(0) * \
            (np.minimum(boxes1[:, None, 3], boxes2[None, :, 3])
             - np.maximum(boxes1[:, None, 1], boxes2[None, :, 1])).clip(0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    return inter / (area2[None, :] + eps)


def copy_paste_seg(img, labels, segments, hyp, rng):
    """官方 flip 模式 CopyPaste：把当前图像水平翻转后，按 p 比例粘贴与现有框 IoA<0.3 的实例

    与官方 `CopyPaste(mode="flip")` 同口径（非 AGPL 代码移植，按行为重实现）：
    - 待粘贴实例 = 当前实例水平翻转（同一张图的镜像）；
    - 与现有框的最大 IoA < 0.30 的实例才可选，按最大 IoA 升序取前 round(p * N) 个；
    - 用选中实例的多边形生成粘贴掩码，从翻转图像复制像素；
    - 标签/多边形追加选中实例。
    """
    if not len(segments) or not len(labels) or hyp.copy_paste <= 0:
        return img, labels, segments
    H, W = img.shape[:2]
    flipped = _flip_lr_segments(segments, W)
    boxes2 = np.stack([polygon_bbox(p) for p in flipped]).astype(np.float32)
    ioa = _bbox_ioa(boxes2, labels[:, 1:5])
    eligible = np.nonzero((ioa < 0.30).all(1))[0]
    if not len(eligible):
        return img, labels, segments
    max_ioa = ioa.max(1) if ioa.shape[1] else np.zeros(len(boxes2), np.float32)
    order = eligible[np.argsort(max_ioa[eligible])]
    n_sel = int(round(hyp.copy_paste * len(order)))
    selected = order[:n_sel]
    if not len(selected):
        return img, labels, segments

    paste = np.zeros((H, W), np.uint8)
    for j in selected:
        cv2.fillPoly(paste, [np.asarray(flipped[j], np.int32).reshape(-1, 1, 2)], 1)
    src = img[:, ::-1]
    img = np.where(paste[..., None].astype(bool), src, img)
    labels = np.concatenate([labels, labels[selected]], axis=0).astype(np.float32)
    return img, labels, segments + [flipped[j] for j in selected]


def copy_paste(img, labels, img_loader, n, hyp, rng):
    """框级 CopyPaste 近似（copy_paste_mode="box" 时启用；官方检测口径为 no-op，见模块 docstring）

    把另一张图随机子集**框矩形**粘贴到当前画布（不做多边形光栅化）：贴入区域覆盖原图内容，
    被覆盖对象的标签仍然保留 —— 属已知的标签噪声来源，非官方行为，仅在显式开启时使用。
    """
    other_img, other_labels = img_loader(int(rng.integers(0, n)))
    H, W = img.shape[:2]
    other_img = cv2.resize(other_img, (W, H), interpolation=cv2.INTER_LINEAR)
    if not len(other_labels):
        return img, labels
    oh, ow = other_img.shape[:2]
    other_labels = other_labels.copy()
    other_labels[:, 1::2] *= W / ow
    other_labels[:, 2::2] *= H / oh

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


def _random_perspective(img, labels, hyp, rng, segments):
    S = hyp.imgsz
    h, w = img.shape[:2]
    angle = float(rng.uniform(-hyp.degrees, hyp.degrees))
    # 缩放增益 g ~ U(1-aug_scale, 1+aug_scale)（官方 random_perspective 口径：
    # 0.5 -> [0.5, 1.5]；曾用 [s, 1/s] = [0.5, 2.0]，上采样端偏强，见 CHANGELOG）
    g = float(rng.uniform(1 - hyp.aug_scale, 1 + hyp.aug_scale))
    shear_x = float(np.tan(rng.uniform(-hyp.shear, hyp.shear) * np.pi / 180))
    shear_y = float(np.tan(rng.uniform(-hyp.shear, hyp.shear) * np.pi / 180))
    tx = float(rng.uniform(0.5 - hyp.translate, 0.5 + hyp.translate)) * S
    ty = float(rng.uniform(0.5 - hyp.translate, 0.5 + hyp.translate)) * S
    # 透视项只在开启时抽样：perspective=0 不消耗 RNG，历史 run 的随机流逐位不变
    px = py = 0.0
    if hyp.perspective:
        px = float(rng.uniform(-hyp.perspective, hyp.perspective))
        py = float(rng.uniform(-hyp.perspective, hyp.perspective))

    C = np.eye(3, dtype=np.float32)  # 画布中心 -> 原点
    C[0, 2], C[1, 2] = -w / 2, -h / 2
    R = np.eye(3, dtype=np.float32)  # 旋转 + 缩放（绕原点 = 画布中心）
    R[:2] = cv2.getRotationMatrix2D(angle=angle, center=(0, 0), scale=g)
    Sh = np.eye(3, dtype=np.float32)  # x/y 双向剪切（官方 S，作用于旋转后）
    Sh[0, 1], Sh[1, 0] = shear_x, shear_y
    T = np.eye(3, dtype=np.float32)  # 平移到输出画布 (0.5±translate)*S
    T[0, 2], T[1, 2] = tx, ty
    P = np.eye(3, dtype=np.float32)  # 透视（官方 P[2,0]=x / P[2,1]=y）
    P[2, 0], P[2, 1] = px, py
    M = T @ Sh @ R @ P @ C  # 官方顺序 T @ S @ R @ P @ C

    if hyp.perspective:
        out = cv2.warpPerspective(img, M, (S, S), borderValue=(BORDER,) * 3)
    else:
        out = cv2.warpAffine(img, M[:2], (S, S), borderValue=(BORDER,) * 3)
    pre = labels[:, 1:5] * g if len(labels) else np.zeros((0, 4), np.float32)  # 官方 box1

    if segments is None:
        if not len(labels):
            return out, labels, None
        lbs = labels.copy()
        # 角点变换 -> 外接框
        corners = np.stack(
            [lbs[:, 1:3], np.stack([lbs[:, 3], lbs[:, 2]], 1), lbs[:, 3:5], np.stack([lbs[:, 1], lbs[:, 4]], 1)], 1
        )  # (M, 4, 2)
        new = _transform_boxes(corners, M, bool(hyp.perspective))
        keep = _box_candidates(pre, _clip_boxes(new, S, S))
        return out, np.concatenate([lbs[:, :1], new], 1)[keep], None

    # 分割：多边形逐点仿射 -> clip -> 框由多边形包围盒导出（官方 area_thr=0.01）
    polys = _clip_segments(_affine_segments(segments, M, bool(hyp.perspective)), S, S)
    boxes = (np.stack([polygon_bbox(p) for p in polys]).astype(np.float32)
             if polys else np.zeros((0, 4), np.float32))
    keep = _box_candidates(pre, boxes, area_thr=0.01) if len(boxes) else np.zeros(0, bool)
    lbs = np.concatenate([labels[:, :1], boxes], 1)[keep] if len(labels) else labels
    return out, lbs.astype(np.float32), _subset_segments(polys, keep)


def random_perspective(img, labels, hyp, rng, segments=None):
    """随机仿射/透视（旋转/各向同性缩放/x-y 剪切/透视/平移）-> S×S

    M = T @ S @ R @ P @ C（官方 _compute_affine_matrix）；perspective=0 时 P=I、
    走 warpAffine 且不抽样（历史随机流逐位不变），>0 时走 warpPerspective。
    旋转缩放绕当前画布中心，平移量以**输出尺寸** S 为单位。
    segments 给定时返回三元组；框由变换后的多边形包围盒导出，过滤用 area_thr=0.01。
    """
    return _ret(*_random_perspective(img, labels, hyp, rng, segments))


def mixup(img, labels, other_img, other_labels, alpha, rng, segments=None, other_segments=None):
    """图像混合：img*(1-r) + other*r，r ~ Beta(alpha, alpha)；标签（与多边形）拼接"""
    r = float(rng.beta(alpha, alpha))
    img = np.clip(img * (1 - r) + other_img * r, 0, 255).astype(np.uint8)
    if len(labels) and len(other_labels):
        labels = np.concatenate([labels, other_labels])
        if segments is not None:
            segments = segments + other_segments
    elif len(other_labels):
        labels = other_labels
        if segments is not None:
            segments = other_segments
    return _ret(img, labels, segments)


def random_hsv(img, h_gain, s_gain, v_gain, rng):
    """HSV 空间随机增益（官方 RandomHSV）

    hue 加性偏移 (x + r0*180) % 180；sat/val 乘性增益并 clip 到 255；
    lut_sat[0] = 0（纯白不改色，官方 8.3.79 起）。
    """
    if not (h_gain or s_gain or v_gain):
        return img
    r = rng.uniform(-1, 1, 3) * [h_gain, s_gain, v_gain]
    x = np.arange(0, 256, dtype=r.dtype)
    lut_hue = ((x + r[0] * 180) % 180).astype(np.uint8)
    lut_sat = np.clip(x * (r[1] + 1), 0, 255).astype(np.uint8)
    lut_val = np.clip(x * (r[2] + 1), 0, 255).astype(np.uint8)
    lut_sat[0] = 0
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def flip_lr(img, labels, segments=None):
    """水平镜像：x' = W - x"""
    out = img[:, ::-1].copy()
    lbs = labels.copy()
    W = img.shape[1]
    if len(lbs):
        x1 = lbs[:, 1].copy()
        lbs[:, 1] = W - lbs[:, 3]
        lbs[:, 3] = W - x1
    segs = _flip_lr_segments(segments, W) if segments is not None else None
    return _ret(out, lbs, segs)


def flip_ud(img, labels, segments=None):
    """垂直镜像：y' = H - y"""
    out = img[::-1].copy()
    lbs = labels.copy()
    H = img.shape[0]
    if len(lbs):
        y1 = lbs[:, 2].copy()
        lbs[:, 2] = H - lbs[:, 4]
        lbs[:, 4] = H - y1
    segs = _flip_ud_segments(segments, H) if segments is not None else None
    return _ret(out, lbs, segs)


def bgr_swap(img):
    """BGR <-> RGB 通道序互换（官方 Format 的 bgr 分支：`img[::-1]`）"""
    return img[:, :, ::-1].copy()


def format_img(img_bgr):
    """BGR uint8 -> RGB float32 [0,1] (3,H,W) 张量"""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    t = np.ascontiguousarray(rgb.transpose(2, 0, 1)) / 255.0
    return torch.from_numpy(t.astype(np.float32))


def augment(img, labels, img_loader, hyp, rng, idx, n, segments=None):
    """完整训练增强 -> (tensor (3,S,S), labels (M,5)[, masks])

    segments 给定时（分割训练）返回三元组；最后一个算子之后统一栅格化 GT 掩码
    （overlap 时按面积降序重排 labels，与实例索引图 ID 对齐）。
    """
    S = hyp.imgsz
    seg_on = segments is not None
    if seg_on and len(segments) != len(labels):
        raise ValueError(f"segments/labels misaligned: {len(segments)} vs {len(labels)}")
    if rng.random() < hyp.mosaic:
        img, labels, segments = _mosaic(img_loader, idx, n, hyp, rng, segments)
    else:
        img, labels, segments = _resize_long_side(img, labels, segments, S)
    if seg_on:
        if hyp.copy_paste > 0 and hyp.copy_paste_mode == "flip":
            img, labels, segments = copy_paste_seg(img, labels, segments, hyp, rng)
    elif hyp.copy_paste > 0 and hyp.copy_paste_mode == "box":
        img, labels = copy_paste(img, labels, img_loader, n, hyp, rng)
    img, labels, segments = _random_perspective(img, labels, hyp, rng, segments)

    if rng.random() < hyp.mixup:
        other_img, other_labels, other_segs = _load_sample(
            img_loader, int(rng.integers(0, n)), seg_on)
        other_img, other_labels, other_segs = _resize_long_side(other_img, other_labels, other_segs, S)
        other_img, other_labels, other_segs = _random_perspective(
            other_img, other_labels, hyp, rng, other_segs)
        if seg_on:
            img, labels, segments = mixup(img, labels, other_img, other_labels, alpha=32.0, rng=rng,
                                          segments=segments, other_segments=other_segs)
        else:
            img, labels = mixup(img, labels, other_img, other_labels, alpha=32.0, rng=rng)

    img = random_hsv(img, hyp.hsv_h, hyp.hsv_s, hyp.hsv_v, rng)
    if rng.random() < hyp.flipud:
        if seg_on:
            img, labels, segments = flip_ud(img, labels, segments=segments)
        else:
            img, labels = flip_ud(img, labels)
    if rng.random() < hyp.fliplr:
        if seg_on:
            img, labels, segments = flip_lr(img, labels, segments=segments)
        else:
            img, labels = flip_lr(img, labels)
    if rng.random() < hyp.bgr:
        img = bgr_swap(img)
    if not seg_on:
        return format_img(img), labels

    masks, order = rasterize_segments(segments, S, S, hyp.mask_ratio, hyp.overlap_mask)
    if order is not None:  # overlap：标签按掩码面积降序重排，与索引 ID 一致
        labels = labels[order]
    return format_img(img), labels, masks


def letterbox_train(img, labels, imgsz, segments=None, mask_ratio=4, overlap_mask=True):
    """augment=False 路径：等比缩放 + 灰边（复用 data/preprocess.letterbox 口径）"""
    from data.preprocess import letterbox

    img, ratio, (pad_top, pad_left) = letterbox(img, new_shape=imgsz)
    lbs = labels.copy()
    if len(lbs):
        lbs[:, 1::2] = lbs[:, 1::2] * ratio + pad_left
        lbs[:, 2::2] = lbs[:, 2::2] * ratio + pad_top
        lbs = _clip(lbs, imgsz, imgsz)
    if segments is None:
        return format_img(img), _drop_zero_area(lbs)
    segs = _shift_segments(_scale_segments(segments, ratio, ratio), pad_left, pad_top)
    keep = _zero_area_keep(lbs)
    lbs = lbs[keep]
    segs = _clip_segments(_subset_segments(segs, keep), imgsz, imgsz)
    masks, order = rasterize_segments(segs, imgsz, imgsz, mask_ratio, overlap_mask)
    if order is not None:
        lbs = lbs[order]
    return format_img(img), lbs, masks
