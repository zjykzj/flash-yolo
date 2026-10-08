"""权重读写：safetensors <-> 模型（strict load，成功即证明拓扑一致）"""

import logging
import re
from pathlib import Path

from safetensors.torch import load_file, save_file

__all__ = ["load_weights", "save_weights", "scale_from_weights", "arch_from_weights", "resolve_arch_scale"]

logger = logging.getLogger(__name__)

# 官方档位命名：yolo26n/s/m/l/x（scripts/download_weights.py 与 scripts/export.py 都沿用）
_SCALE_RE = re.compile(r"yolo26([nsmlx])($|[-_.])", re.I)
# 架构命名：yolov3-tiny（官方 darknet 权重命名）
_ARCH_RE = re.compile(r"(yolov3-tiny)($|[-_.])", re.I)


def scale_from_weights(path):
    """从权重文件名推模型档位：`yolo26s.safetensors` / `yolo26s.onnx` -> "s"；认不出返回 None

    推理脚本据此不必强制 `--scale`：官方命名自带档位。认不出（如 `best.safetensors`）时由调用方
    报错要求显式指定——比闷头按 n 档建模型、再到 strict load 处报尺寸不匹配要好。
    """
    m = _SCALE_RE.search(Path(path).stem)
    return m.group(1).lower() if m else None


def arch_from_weights(path):
    """从权重文件名推架构：`yolov3-tiny.safetensors` -> "yolov3-tiny"；认不出返回 None"""
    m = _ARCH_RE.search(Path(path).stem)
    return m.group(1).lower() if m else None


def resolve_arch_scale(weights, model=None, scale=None):
    """(arch, scale) 解析：显式参数优先，缺省从权重文件名推断；都认不出返回 (None, None)

    yolo26 系：`yolo26s.safetensors` -> ("yolo26", "s")（档位认不出时 scale=None，调用方报错）；
    v3 系：`yolov3-tiny.safetensors` -> ("yolov3-tiny", None)（档位由架构固定，build_model 用默认档）。
    """
    if model is None:
        # 文件名里的架构优先（yolov3-tiny 命名 + 误传的 --scale 不应把它当 yolo26）
        model = arch_from_weights(weights) or ("yolo26" if (scale or scale_from_weights(weights)) else None)
    if model is None:
        return None, None
    if model == "yolo26":
        return model, scale or scale_from_weights(weights)
    return model, None  # 非 yolo26 架构：档位固定（--scale 不适用）


def load_weights(model, path, strict=True):
    """加载转换后的纯权重（由 scripts/convert_weights.py 生成）

    Args:
        model: YOLO26 模型
        path: .safetensors 路径
        strict: True 时任何 key 不匹配都会报错（官方语义：加载成功即拓扑一致）
    """
    sd = load_file(path)
    missing, unexpected = model.load_state_dict(sd, strict=strict)
    if not strict and (missing or unexpected):
        logger.warning(f"missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")
        if missing:
            logger.warning(f"missing sample: {missing[:5]}")
        if unexpected:
            logger.warning(f"unexpected sample: {unexpected[:5]}")
    return missing, unexpected


def save_weights(model, path):
    """模型 state_dict -> safetensors（纯权重交付物，配合 load_weights strict 回载）

    保存前统一转连续：channels_last 训练时参数为 NHWC 步长，safetensors 只接受连续
    张量（仅规范化内存布局，数值不变；回载时按目标参数布局 copy，不受影响）。
    """
    save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}, path)
