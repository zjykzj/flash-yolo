"""实例分割标注：多边形解析 / 变换 / 栅格化（COCO 训练与验证共用，纯 numpy + cv2）

口径对齐官方 ultralytics（`convert_coco` + `Format`）：
- `segmentation` 多边形列表 -> 每段校验偶数长度、>=3 点、数值有限；多段按最近端点连接成一条；
- RLE dict / 缺失 / 无效 -> 退回 bbox 矩形多边形（官方 converter 同款）；调用方统计 fallback；
- 栅格化在最终 letterbox 分辨率 `cv2.fillPoly`，再 `cv2.resize(INTER_LINEAR)` 到 `1/mask_ratio`；
- overlap=True 输出每图一张实例索引图（面积降序，ID 1..M），并返回重排顺序供标签同步重排；
  overlap=False 输出每实例一张二值掩码（保持输入顺序）。

本模块不做语义裁剪：调用方负责把多边形变换到目标坐标系后再栅格化。
"""

import cv2
import numpy as np

__all__ = [
    "annotation_polygon", "merge_polygons", "polygon_bbox", "rectangle_from_bbox",
    "transform_polygons", "clip_polygons", "rasterize_segments",
]


def _valid_part(part):
    """单个多边形：偶数长度、>=3 点、全数值有限"""
    if not isinstance(part, (list, tuple)) or len(part) < 6 or len(part) % 2:
        return None
    try:
        arr = np.asarray(part, dtype=np.float32)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(arr).all():
        return None
    return arr.reshape(-1, 2)


def merge_polygons(parts):
    """多段多边形 -> 一条（最近端点连接；等价官方 merge_multi_segment 的语义）

    反复把下一条多边形的最近端点接到当前折线末端；必要时反转折线/多边形。
    """
    if len(parts) == 1:
        return parts[0]
    merged = parts[0].copy()
    remaining = [p.copy() for p in parts[1:]]
    while remaining:
        best = None  # (dist2, part_idx, current_end, part_end)
        for pi, poly in enumerate(remaining):
            for ce, cp in ((0, merged[0]), (1, merged[-1])):
                for pe, pp in ((0, poly[0]), (1, poly[-1])):
                    d = float(((cp - pp) ** 2).sum())
                    if best is None or d < best[0]:
                        best = (d, pi, ce, pe)
        _, pi, ce, pe = best
        poly = remaining.pop(pi)
        if ce == 0:  # 接在当前起点 -> 反转当前折线，保持"向后追加"的写法
            merged = merged[::-1]
        if pe == 1:  # 多边形从尾部接入 -> 反转它
            poly = poly[::-1]
        merged = np.concatenate([merged, poly], axis=0)
    return merged


def annotation_polygon(ann):
    """一条 COCO annotation -> (polygon (K,2) float32 | None, used_fallback: bool)

    - 多边形 segmentation：返回合并后的多边形，fallback=False；
    - RLE / 缺失 / 无效：若 bbox 有效，返回 bbox 矩形，fallback=True；
    - bbox 也无效：返回 (None, False)。
    """
    seg = ann.get("segmentation")
    parts = []
    if isinstance(seg, (list, tuple)):
        for part in seg:
            poly = _valid_part(part)
            if poly is not None:
                parts.append(poly)
        if parts:
            merged = merge_polygons(parts)
            if len(merged) >= 3:
                return merged.astype(np.float32), False
    bbox = ann.get("bbox")
    if bbox is not None and len(bbox) >= 4:
        x, y, w, h = (float(v) for v in bbox[:4])
        if w > 0 and h > 0 and np.isfinite([x, y, w, h]).all():
            return rectangle_from_bbox((x, y, w, h)), True
    return None, False


def rectangle_from_bbox(bbox):
    """(x, y, w, h) -> 矩形多边形 (4,2) float32（TL, TR, BR, BL）"""
    x, y, w, h = (float(v) for v in bbox[:4])
    return np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], np.float32)


def polygon_bbox(poly):
    """多边形 -> xyxy（像素）；退化（零宽/零高）由调用方过滤"""
    x1, y1 = poly.min(0)
    x2, y2 = poly.max(0)
    return np.array([x1, y1, x2, y2], np.float32)


def transform_polygons(polygons, ratio, pad):
    """letterbox：先等比缩放，再加 (left, top) 偏移；返回新列表"""
    out = []
    for poly in polygons:
        p = poly * ratio
        p = p + np.array([pad[1], pad[0]], np.float32)  # pad = (top, left)
        out.append(p.astype(np.float32))
    return out


def clip_polygons(polygons, w, h):
    """裁剪到 [0, w] x [0, h]（与框的 clip 口径一致）"""
    return [np.stack([np.clip(p[:, 0], 0, w), np.clip(p[:, 1], 0, h)], 1).astype(np.float32)
            for p in polygons]


def rasterize_segments(segments, h, w, mask_ratio=4, overlap=True):
    """多边形（目标坐标系，h×w）-> GT 掩码

    Args:
        segments: list[(K,2) float32]，顺序与标签行一致
        h, w: 当前图像（letterbox 后）高宽
        mask_ratio: 下采样倍率（官方默认 4；4 = Proto26 分辨率）
        overlap: True -> (1, h/r, w/r) int32 实例索引图（面积降序，1..M）；
                 False -> (M, h/r, w/r) uint8 二值掩码（保持输入顺序）

    Returns:
        masks, order: overlap 时 order 为面积降序的原索引（标签需按它重排）；
                     非 overlap 时 order=None。
    """
    if mask_ratio < 1 or h % mask_ratio or w % mask_ratio:
        raise ValueError(f"mask_ratio={mask_ratio} must divide image size {h}x{w}")

    mh, mw = h // mask_ratio, w // mask_ratio
    masks = []
    for poly in segments:
        full = np.zeros((h, w), np.uint8)
        if len(poly) >= 3:
            pts = np.round(poly).astype(np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(full, [pts], 1)
        masks.append(cv2.resize(full, (mw, mh), interpolation=cv2.INTER_LINEAR))

    if not masks:
        if overlap:
            return np.zeros((1, mh, mw), np.int32), np.zeros(0, np.int64)
        return np.zeros((0, mh, mw), np.uint8), None

    if not overlap:
        return np.stack(masks).astype(np.uint8), None

    areas = np.array([int(m.sum()) for m in masks], np.int64)
    order = np.argsort(-areas, kind="stable")
    out = np.zeros((1, mh, mw), np.int32)
    for rank, idx in enumerate(order):
        out[0] = np.maximum(out[0], masks[idx].astype(np.int32) * (rank + 1))
    return out, order
