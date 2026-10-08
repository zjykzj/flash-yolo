"""权重读写：safetensors <-> 模型（strict load，成功即证明拓扑一致）"""

import logging
import re
from pathlib import Path

from safetensors.torch import load_file, save_file

__all__ = ["load_weights", "save_weights", "scale_from_weights"]

logger = logging.getLogger(__name__)

# 官方档位命名：yolo26n/s/m/l/x（scripts/download_weights.py 与 scripts/export.py 都沿用）
_SCALE_RE = re.compile(r"yolo26([nsmlx])($|[-_.])", re.I)


def scale_from_weights(path):
    """从权重文件名推模型档位：`yolo26s.safetensors` / `yolo26s.onnx` -> "s"；认不出返回 None

    推理脚本据此不必强制 `--scale`：官方命名自带档位。认不出（如 `best.safetensors`）时由调用方
    报错要求显式指定——比闷头按 n 档建模型、再到 strict load 处报尺寸不匹配要好。
    """
    m = _SCALE_RE.search(Path(path).stem)
    return m.group(1).lower() if m else None


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
