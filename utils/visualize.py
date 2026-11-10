"""结果可视化：画框 + 类别标签（分割结果额外叠加掩码）"""

from pathlib import Path

import cv2
import numpy as np

__all__ = ["draw_detections", "draw_target_grid", "COCO_SKELETON"]

# COCO 17 点骨架（0-based；官方 json 的 skeleton 是 1-based，描述符/这里统一 0-based）
COCO_KPT_COUNT = 17
COCO_SKELETON = (
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), (5, 6),
    (5, 7), (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6),
)

# 固定调色板（BGR），按类别 id 取色
_COLORS = [
    (255, 128, 0), (255, 153, 51), (255, 178, 102), (230, 230, 0), (255, 153, 255),
    (153, 204, 255), (255, 102, 255), (255, 51, 255), (102, 178, 255), (51, 153, 255),
    (255, 153, 153), (255, 102, 102), (255, 51, 51), (153, 255, 153), (102, 255, 102),
    (51, 255, 51), (0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 255),
    (204, 102, 0), (255, 0, 0), (102, 204, 0), (255, 255, 0), (0, 0, 153),
    (0, 0, 204), (255, 51, 153), (0, 204, 0), (0, 0, 0), (255, 128, 0),
    (255, 255, 51), (153, 153, 0), (153, 51, 153), (102, 51, 0), (51, 51, 255),
    (51, 255, 51), (255, 204, 0), (255, 153, 0), (255, 102, 0), (255, 51, 0),
    (102, 255, 153), (51, 255, 102), (0, 255, 51), (255, 0, 153), (204, 51, 0),
    (255, 204, 153), (255, 153, 102), (255, 102, 51), (255, 51, 0), (153, 255, 255),
    (102, 255, 255), (51, 255, 255), (0, 255, 255), (255, 255, 153), (255, 255, 102),
    (255, 255, 51), (255, 255, 0), (153, 204, 255), (102, 178, 255), (51, 153, 255),
    (0, 128, 255), (153, 255, 153), (102, 255, 102), (51, 255, 51), (0, 255, 0),
    (255, 0, 51), (255, 0, 0), (204, 255, 204), (102, 102, 0), (51, 102, 0),
    (0, 204, 204), (255, 255, 255), (153, 153, 153), (51, 51, 51), (0, 0, 0),
]


def draw_detections(image_bgr, detections, names=None, save_path=None, skeleton=None):
    """在原图上画检测框（Detections.masks 存在时先叠加半透明掩码，再画框与标签）

    Args:
        image_bgr: (H, W, 3)
        detections: Detections（原图坐标；masks 为 (N, H, W) uint8 @原图分辨率；
            keypoints 为 (N, K, D) 时叠加关键点/骨架）
        names: dict[int, str] 类别名（默认 config.datasets.load_names()，即 COCO 80 类）
        save_path: 若给定则保存
        skeleton: 骨架点对（0-based）；None 且 K=17 时用 COCO 骨架；K≠17 时只画点
    """
    if names is None:
        from config.datasets import load_names

        names = load_names()
    out = image_bgr.copy()
    if getattr(detections, "masks", None) is not None:
        out = _draw_masks(out, detections)
    for box, score, cid in zip(detections.boxes, detections.scores, detections.class_ids):
        color = _COLORS[int(cid) % len(_COLORS)]
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{names.get(int(cid), cid)} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, max(y1 - th - 4, 0)), (x1 + tw + 2, y1), color, -1)
        cv2.putText(out, label, (x1 + 1, max(y1 - 3, th)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    kpts = getattr(detections, "keypoints", None)
    if kpts is not None and len(kpts):
        if skeleton is None:
            skeleton = COCO_SKELETON if kpts.shape[1] == COCO_KPT_COUNT else ()
        _draw_keypoints(out, detections, skeleton or ())
    if save_path:
        cv2.imwrite(save_path, out)
    return out


def _draw_keypoints(image_bgr, detections, skeleton):
    """逐实例画骨架线 + 关键点（可见性 <0.5 的点不画；两端都可见才连线）"""
    for kpts, cid in zip(detections.keypoints, detections.class_ids):
        color = _COLORS[int(cid) % len(_COLORS)]
        visible = kpts[:, 2] >= 0.5 if kpts.shape[1] >= 3 else np.ones(len(kpts), bool)
        for i, j in skeleton:
            if i < len(kpts) and j < len(kpts) and visible[i] and visible[j]:
                p1 = (int(round(kpts[i, 0])), int(round(kpts[i, 1])))
                p2 = (int(round(kpts[j, 0])), int(round(kpts[j, 1])))
                cv2.line(image_bgr, p1, p2, color, 1, cv2.LINE_AA)
        for q, (x, y) in enumerate(kpts[:, :2]):
            if visible[q]:
                cv2.circle(image_bgr, (int(round(x)), int(round(y))), 3, color, -1, cv2.LINE_AA)
                cv2.circle(image_bgr, (int(round(x)), int(round(y))), 3, (0, 0, 0), 1, cv2.LINE_AA)


def _draw_masks(image_bgr, detections, alpha=0.5):
    """按类别色叠加半透明掩码（后画的覆盖先画的——实例重叠时与画框顺序一致）"""
    out = image_bgr.copy().astype(np.float32)
    for mask, cid in zip(detections.masks, detections.class_ids):
        m = mask.astype(bool)
        if not m.any():
            continue
        color = np.array(_COLORS[int(cid) % len(_COLORS)], dtype=np.float32)
        out[m] = out[m] * (1 - alpha) + color * alpha
    return out.astype(np.uint8)


def draw_target_grid(imgs, targets, save_path, n=8, names=None, scale=1.0,
                     keypoints=None, skeleton=None):
    """训练 batch 抽样网格图：在增强后的图上画 GT 框（pose 另画关键点）并拼接保存

    Args:
        imgs: (B,3,H,W) float32 [0,1] RGB 张量（训练输入，已增强，可含 mosaic/mixup）
        targets: (N,6) [batch_idx, cls, x1, y1, x2, y2] 当前画布像素坐标
        save_path: 保存路径（PNG）
        n: 抽样张数（取 batch 前 n 张）
        names: 类别名（默认 load_names()，即 COCO 80 类）
        scale: 单张缩放（0.5 = 半尺寸，控文件大小）
        keypoints: (N,K,D) 与 targets 行对齐的 GT 关键点（pose；可见性 <0.5 不画）
        skeleton: 骨架点对（0-based）；None 且 K=17 时用 COCO 骨架
    """
    if names is None:
        from config.datasets import load_names

        names = load_names()
    tgt = targets.detach().cpu().numpy() if hasattr(targets, "detach") else np.asarray(targets)
    kps = (keypoints.detach().cpu().numpy() if hasattr(keypoints, "detach") else
           np.asarray(keypoints) if keypoints is not None else None)
    if kps is not None and skeleton is None:
        skeleton = COCO_SKELETON if kps.shape[1] == COCO_KPT_COUNT else ()
    tiles = []
    b = int(imgs.shape[0])
    for i in range(min(n, b)):
        img = imgs[i].detach().cpu().numpy().transpose(1, 2, 0)  # RGB (H,W,3) [0,1]
        img = np.ascontiguousarray(img[:, :, ::-1])  # -> BGR
        img = (img * 255).clip(0, 255).astype(np.uint8)
        rows = tgt[tgt[:, 0] == i]
        rows_kp = kps[tgt[:, 0] == i] if kps is not None else [None] * len(rows)
        for row, kp in zip(rows, rows_kp):
            x1, y1, x2, y2 = map(int, row[2:6])
            color = _COLORS[int(row[1]) % len(_COLORS)]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label = names.get(int(row[1]), str(int(row[1])))
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (max(x1 - tw - 2, 0), max(y1 - th - 4, 0)), (x1, y1), color, -1)
            cv2.putText(img, label, (max(x1 - tw, 0), max(y1 - 3, th)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            if kp is not None:
                visible = kp[:, 2] >= 0.5 if kp.shape[1] >= 3 else np.ones(len(kp), bool)
                for a, b_ in skeleton or ():
                    if visible[a] and visible[b_]:
                        cv2.line(img, (int(round(kp[a, 0])), int(round(kp[a, 1]))),
                                 (int(round(kp[b_, 0])), int(round(kp[b_, 1]))), color, 1, cv2.LINE_AA)
                for q, (x, y) in enumerate(kp[:, :2]):
                    if visible[q]:
                        cv2.circle(img, (int(round(x)), int(round(y))), 2, color, -1, cv2.LINE_AA)
        if scale != 1.0:
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        tiles.append(img)
    if not tiles:
        return None
    cols = int(np.ceil(np.sqrt(len(tiles))))
    rows = int(np.ceil(len(tiles) / cols))
    h, w = tiles[0].shape[:2]
    grid = np.full((rows * h, cols * w, 3), 114, np.uint8)
    for k, tile in enumerate(tiles):
        r, c = divmod(k, cols)
        grid[r * h : r * h + tile.shape[0], c * w : c * w + tile.shape[1]] = tile
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(save_path), grid):
        raise OSError(f"cannot write augment sample grid: {save_path}")
    return grid
