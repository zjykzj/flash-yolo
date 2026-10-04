"""检测结果可视化：画框 + 类别标签"""

import cv2

__all__ = ["draw_detections"]

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
