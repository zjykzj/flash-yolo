"""检测结果可视化：画框 + 类别标签"""

from pathlib import Path

import cv2
import numpy as np

__all__ = ["draw_detections", "draw_target_grid"]

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


def draw_detections(image_bgr, detections, names=None, save_path=None):
    """在原图上画检测框

    Args:
        image_bgr: (H, W, 3)
        detections: Detections（原图坐标）
        names: dict[int, str] 类别名（默认用 config.defaults.COCO_NAMES）
        save_path: 若给定则保存
    """
    if names is None:
        from config.defaults import COCO_NAMES

        names = COCO_NAMES
    out = image_bgr.copy()
    for box, score, cid in zip(detections.boxes, detections.scores, detections.class_ids):
        color = _COLORS[int(cid) % len(_COLORS)]
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{names.get(int(cid), cid)} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, max(y1 - th - 4, 0)), (x1 + tw + 2, y1), color, -1)
        cv2.putText(out, label, (x1 + 1, max(y1 - 3, th)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    if save_path:
        cv2.imwrite(save_path, out)
    return out


def draw_target_grid(imgs, targets, save_path, n=8, names=None, scale=1.0):
    """训练 batch 抽样网格图：在增强后的图上画 GT 框并拼接保存

    Args:
        imgs: (B,3,H,W) float32 [0,1] RGB 张量（训练输入，已增强，可含 mosaic/mixup）
        targets: (N,6) [batch_idx, cls, x1, y1, x2, y2] 当前画布像素坐标
        save_path: 保存路径（PNG）
        n: 抽样张数（取 batch 前 n 张）
        names: 类别名（默认 COCO_NAMES）
        scale: 单张缩放（0.5 = 半尺寸，控文件大小）
    """
    if names is None:
        from config.defaults import COCO_NAMES

        names = COCO_NAMES
    tgt = targets.detach().cpu().numpy() if hasattr(targets, "detach") else np.asarray(targets)
    tiles = []
    b = int(imgs.shape[0])
    for i in range(min(n, b)):
        img = imgs[i].detach().cpu().numpy().transpose(1, 2, 0)  # RGB (H,W,3) [0,1]
        img = np.ascontiguousarray(img[:, :, ::-1])  # -> BGR
        img = (img * 255).clip(0, 255).astype(np.uint8)
        for row in tgt[tgt[:, 0] == i]:
            x1, y1, x2, y2 = map(int, row[2:6])
            color = _COLORS[int(row[1]) % len(_COLORS)]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label = names.get(int(row[1]), str(int(row[1])))
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (max(x1 - tw - 2, 0), max(y1 - th - 4, 0)), (x1, y1), color, -1)
            cv2.putText(img, label, (max(x1 - tw, 0), max(y1 - 3, th)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
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
