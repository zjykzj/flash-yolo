"""训练损失：按「结构家族 × 任务」分文件（规则同 model/head/__init__.py）

家族短名对照：
    yolo26_*  ->  yolo26 / yolo26-seg / flash-yolo（flash-yolo 复用 yolo26 检测损失）
    v3_*      ->  yolov3-tiny / yolov3-tiny-silu

本模块只做分派与公共导出；实现见同目录模块。接口同构：item_keys / __call__。
"""

from train.loss.v3_detect import ComputeLossV3
from train.loss.yolo26_detect import ComputeLoss
from train.loss.yolo26_seg import ComputeLossSeg

__all__ = ["ComputeLoss", "ComputeLossSeg", "ComputeLossV3", "build_loss"]

_LOSS_BY_ARCH = {
    "yolo26": ComputeLoss,
    "flash-yolo": ComputeLoss,
    "yolo26-seg": ComputeLossSeg,
    "flash-yolo-seg": ComputeLossSeg,
    "yolov3-tiny": ComputeLossV3,
    "yolov3-tiny-silu": ComputeLossV3,
}


def build_loss(arch, cfg, head, device):
    """按架构分派损失实现（trainer 的单一入口）"""
    cls = _LOSS_BY_ARCH.get(arch)
    if cls is None:
        raise ValueError(f"unknown arch {arch!r}; available: {sorted(_LOSS_BY_ARCH)}")
    return cls(cfg, head, device)
