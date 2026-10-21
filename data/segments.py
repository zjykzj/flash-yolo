"""实例分割标注：多边形解析 / 变换 / 栅格化（COCO 训练与验证共用，纯 numpy + cv2）

按 Apache-2.0 项目约束：以下为按官方 converter/utils 的行为独立重写，未复制 AGPL 源码。

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
    "annotation_polygon", "annotation_polygon_ex", "decode_rle", "rle_polygon", "merge_polygons",
    "polygon_bbox", "rectangle_from_bbox", "transform_polygons", "clip_polygons",
    "rasterize_segments", "rasterize_segments_multi",
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


def _closest_pair(a, b):
    """两组点中距离最近的一对下标（官方 min_index 口径）"""
    d = ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)
    return np.unravel_index(np.argmin(d), d.shape)


def merge_polygons(parts):
    """多段多边形 -> 一条（按官方 merge_multi_segment 的两轮连接口径重实现）

    先把相邻段按最近点对记录连接下标，再正向/反向各走一遍拼成一条折线；
    官方 converter 随后把返回的各段 concatenate，本函数直接返回拼接结果。
    """
    if not parts:
        return np.zeros((0, 2), np.float32)
    if len(parts) == 1:
        return np.asarray(parts[0], np.float32)
    segments = [np.asarray(p, np.float32).reshape(-1, 2) for p in parts]
    idx_list = [[] for _ in segments]
    for i in range(1, len(segments)):
        i1, i2 = _closest_pair(segments[i - 1], segments[i])
        idx_list[i - 1].append(int(i1))
        idx_list[i].append(int(i2))

    paths = []
    for k in range(2):
        if k == 0:  # 正向：每段滚到连接点，首尾段整段保留，中间段取到下一个连接点
            for i, idx in enumerate(idx_list):
                if len(idx) == 2 and idx[0] > idx[1]:
                    idx = idx[::-1]
                    segments[i] = segments[i][::-1]
                segments[i] = np.roll(segments[i], -idx[0], axis=0)
                segments[i] = np.concatenate([segments[i], segments[i][:1]])
                if i in (0, len(idx_list) - 1):
                    paths.append(segments[i])
                else:
                    paths.append(segments[i][: idx[1] - idx[0] + 1])
        else:  # 反向：只补中间段被截掉的尾部
            for i in range(len(idx_list) - 1, -1, -1):
                if i not in (0, len(idx_list) - 1):
                    n = abs(idx_list[i][1] - idx_list[i][0])
                    paths.append(segments[i][n:])
    return np.concatenate(paths, axis=0).astype(np.float32)


def _rle_counts_from_string(s):
    """COCO 压缩 RLE counts 字符串 -> run-length 列表（cocoapi `rleFrString` 口径）

    5 bit 一组、0x20 为续位；差值编码（从第 3 个 count 起累加前前项），
    最高位 0x10 为符号位。非法/截断字符串抛 ValueError，由调用方回退 bbox。
    """
    counts, p, n = [], 0, len(s)
    while p < n:
        x, k, more = 0, 0, 1
        while more:
            if p >= n:
                raise ValueError("truncated RLE counts")
            c = ord(s[p]) - 48
            if not 0 <= c <= 63:
                raise ValueError("invalid RLE counts character")
            x |= (c & 0x1F) << (5 * k)
            more = c & 0x20
            p += 1
            k += 1
            if not more and (c & 0x10):
                x |= -1 << (5 * k)
        if len(counts) > 2:
            x += counts[-2]
        counts.append(x)
    return counts


def decode_rle(segmentation):
    """COCO RLE dict -> (h, w) uint8 掩码（纯 numpy；支持压缩字符串与 counts 列表）

    counts 从背景(0)开始交替，按**列优先**（Fortran order）铺回图像——与 cocoapi 一致。
    结构/长度/字符非法时抛 ValueError，由 `annotation_polygon_ex` 统一回退 bbox。
    """
    if not isinstance(segmentation, dict):
        raise ValueError("RLE segmentation must be a dict")
    size = segmentation.get("size")
    if not isinstance(size, (list, tuple)) or len(size) != 2:
        raise ValueError("RLE 'size' must be [h, w]")
    h, w = int(size[0]), int(size[1])
    if h <= 0 or w <= 0:
        raise ValueError(f"invalid RLE size {h}x{w}")
    counts = segmentation.get("counts")
    if isinstance(counts, str):
        counts = _rle_counts_from_string(counts)
    elif isinstance(counts, (list, tuple)):
        counts = [int(c) for c in counts]
    else:
        raise ValueError("RLE 'counts' must be a string or a list")
    if any(c < 0 for c in counts):
        raise ValueError("negative RLE count")
    if sum(counts) != h * w:
        raise ValueError(f"RLE counts sum {sum(counts)} != {h}x{w}")
    flat = np.zeros(h * w, np.uint8)
    pos = 0
    for i, c in enumerate(counts):
        if i & 1:
            flat[pos:pos + c] = 1
        pos += c
    return flat.reshape((h, w), order="F")


def rle_polygon(segmentation):
    """COCO RLE dict -> 多边形 (K,2) float32 | None（空掩码返回 None）

    RLE 本质是掩码：解码后取外轮廓，再按既有多段合并口径连成一条多边形
    （多连通域会像官方 converter 一样用最近点连接，面积近似而非逐像素等价）。
    轮廓点是像素中心，因此由它导出的框可能比标注 bbox 小 1px；栅格化回掩码与
    RLE 解码结果逐像素一致（单连通域，回归测试覆盖）。
    """
    mask = decode_rle(segmentation)
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    parts = [c.reshape(-1, 2).astype(np.float32) for c in contours if len(c) >= 3]
    if not parts:
        return None
    merged = merge_polygons(parts)
    return merged.astype(np.float32) if len(merged) >= 3 else None


def annotation_polygon_ex(ann):
    """一条 COCO annotation -> (polygon | None, kind)

    kind 取值：
    - "polygon"：`segmentation` 为多边形列表（合并多段）；
    - "rle"：`segmentation` 为 RLE dict 且成功解码（空掩码时 polygon 为 None）；
    - "bbox"：RLE/多边形缺失或非法，退回 bbox 矩形；
    - None：bbox 也无效。
    """
    seg = ann.get("segmentation")
    if isinstance(seg, dict):
        try:
            return rle_polygon(seg), "rle"
        except (ValueError, TypeError, cv2.error):
            pass  # 损坏的 RLE：与缺失 segmentation 同口径，退回 bbox
    parts = []
    if isinstance(seg, (list, tuple)):
        for part in seg:
            poly = _valid_part(part)
            if poly is not None:
                parts.append(poly)
        if parts:
            merged = merge_polygons(parts)
            if len(merged) >= 3:
                return merged.astype(np.float32), "polygon"
    bbox = ann.get("bbox")
    if bbox is not None and len(bbox) >= 4:
        x, y, w, h = (float(v) for v in bbox[:4])
        if w > 0 and h > 0 and np.isfinite([x, y, w, h]).all():
            return rectangle_from_bbox((x, y, w, h)), "bbox"
    return None, None


def annotation_polygon(ann):
    """一条 COCO annotation -> (polygon (K,2) float32 | None, used_fallback: bool)

    兼容旧调用：fallback 仅在退回 bbox 矩形时为 True；RLE 成功解码按正式标注处理。
    """
    poly, kind = annotation_polygon_ex(ann)
    return poly, kind == "bbox"


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


def rasterize_segments_multi(segments, h, w, specs):
    """多边形 -> 多套 GT 掩码（每个实例只 fillPoly 一次，按倍率共用 resize 结果）

    训练中验证同时需要 loss 网格（`mask_ratio` / `overlap_mask`）与指标网格
    （固定 `mask_ratio=4` / `overlap=False`）：此前调用两次会重复逐实例 fillPoly。
    这里按倍率去重后只栅格化一次。

    Args:
        segments: list[(K,2) float32]，顺序与标签行一致
        h, w: 当前图像（letterbox 后）高宽
        specs: list[(mask_ratio, overlap)]；相同 mask_ratio 的输出共用一次 resize

    Returns:
        list[(masks, order)]，顺序与 specs 一致：
        overlap=True  -> (1, h/r, w/r) 实例索引图（面积降序，1..M），order 为面积降序原索引；
        overlap=False -> (M, h/r, w/r) uint8 二值掩码（保持输入顺序），order=None。
    """
    for mask_ratio, _ in specs:
        if mask_ratio < 1 or h % mask_ratio or w % mask_ratio:
            raise ValueError(f"mask_ratio={mask_ratio} must divide image size {h}x{w}")

    ratios = list(dict.fromkeys(int(r) for r, _ in specs))
    grids = {r: (h // r, w // r) for r in ratios}
    per_ratio = {r: [] for r in ratios}  # ratio -> resized masks（输入顺序）
    for poly in segments:
        full = np.zeros((h, w), np.uint8)
        if len(poly) >= 3:
            # 官方 polygon2mask 直接 np.asarray(..., int32)（截断，不四舍五入）
            pts = np.asarray(poly, dtype=np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(full, [pts], 1)
        for r in ratios:
            mh, mw = grids[r]
            per_ratio[r].append(cv2.resize(full, (mw, mh), interpolation=cv2.INTER_LINEAR))

    results = []
    for mask_ratio, overlap in specs:
        r = int(mask_ratio)
        masks = per_ratio[r]
        mh, mw = grids[r]
        if not overlap:
            if not masks:
                results.append((np.zeros((0, mh, mw), np.uint8), None))
            else:
                results.append((np.stack(masks).astype(np.uint8), None))
            continue
        dtype = np.uint8 if len(segments) <= 255 else np.int32
        if not masks:
            results.append((np.zeros((1, mh, mw), dtype), np.zeros(0, np.int64)))
            continue
        areas = np.array([int(m.sum()) for m in masks], np.int64)
        order = np.argsort(-areas, kind="stable")
        out = np.zeros((1, mh, mw), dtype=dtype)
        for rank, idx in enumerate(order):
            out[0] = np.maximum(out[0], masks[idx].astype(dtype) * (rank + 1))
        results.append((out, order))
    return results


def rasterize_segments(segments, h, w, mask_ratio=4, overlap=True):
    """多边形（目标坐标系，h×w）-> 单套 GT 掩码（`rasterize_segments_multi` 的单项包装）

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
    return rasterize_segments_multi(segments, h, w, [(mask_ratio, overlap)])[0]
