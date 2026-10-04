"""权重加载：safetensors -> 模型（strict load，成功即证明拓扑一致）"""

import logging

from safetensors.torch import load_file

__all__ = ["load_weights"]

logger = logging.getLogger(__name__)


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
