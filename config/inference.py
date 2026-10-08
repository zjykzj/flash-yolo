"""推理 / 评估默认值：输入尺寸 · 置信度/NMS 阈值 · 每图最大检测数

训练侧不读本模块：`Trainer` 把 `cfg.imgsz` 显式传给每轮验证（trainer.py -> validator.py），
阈值也各走各的配置；这里只服务推理（infer/engine/postprocess）与评估（eval）链路。
"""

# ---- 推理 / 评估默认超参（与官方 val 口径一致）----
IMGSZ = 640          # 推理输入尺寸
CONF_THRES = 0.001   # 置信度阈值（评估时与官方 val 相同）
IOU_THRES = 0.7      # NMS IoU 阈值（与官方 val 相同）
MAX_DET = 300        # 每图最大检测数

__all__ = ["IMGSZ", "CONF_THRES", "IOU_THRES", "MAX_DET"]
