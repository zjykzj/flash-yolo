"""推理 / 评估默认值：输入尺寸 · 置信度/NMS 阈值 · 每图最大检测数

训练侧不读本模块：`Trainer` 把 `cfg.imgsz` 显式传给每轮验证（trainer.py -> validator.py），
阈值也各走各的配置；这里只服务推理（infer/engine/postprocess）与评估（eval）链路。
"""

# ---- 推理 / 评估默认超参（与官方 val 口径一致）----
IMGSZ = 640          # 推理输入尺寸
CONF_THRES = 0.001   # 置信度阈值（评估时与官方 val 相同）
IOU_THRES = 0.7      # NMS IoU 阈值（与官方 val 相同）
MAX_DET = 300        # 每图最大检测数

__all__ = ["IMGSZ", "CONF_THRES", "IOU_THRES", "MAX_DET", "COCO_NAMES"]

# ---- COCO 80 类 ----
COCO_NAMES = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane",
    5: "bus", 6: "train", 7: "truck", 8: "boat", 9: "traffic light",
    10: "fire hydrant", 11: "stop sign", 12: "parking meter", 13: "bench",
    14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow",
    20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe", 24: "backpack",
    25: "umbrella", 26: "handbag", 27: "tie", 28: "suitcase", 29: "frisbee",
    30: "skis", 31: "snowboard", 32: "sports ball", 33: "kite",
    34: "baseball bat", 35: "baseball glove", 36: "skateboard",
    37: "surfboard", 38: "tennis racket", 39: "bottle", 40: "wine glass",
    41: "cup", 42: "fork", 43: "knife", 44: "spoon", 45: "bowl",
    46: "banana", 47: "apple", 48: "sandwich", 49: "orange",
    50: "broccoli", 51: "carrot", 52: "hot dog", 53: "pizza", 54: "donut",
    55: "cake", 56: "chair", 57: "couch", 58: "potted plant", 59: "bed",
    60: "dining table", 61: "toilet", 62: "tv", 63: "laptop", 64: "mouse",
    65: "remote", 66: "keyboard", 67: "cell phone", 68: "microwave",
    69: "oven", 70: "toaster", 71: "sink", 72: "refrigerator", 73: "book",
    74: "clock", 75: "vase", 76: "scissors", 77: "teddy bear",
    78: "hair drier", 79: "toothbrush",
}
