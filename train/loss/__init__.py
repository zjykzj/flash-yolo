"""训练损失：按「结构家族 × 任务」分文件（规则同 model/head/__init__.py）

家族短名对照：
    yolo26_*  ->  yolo26 / yolo26-seg / yolo26-pose / flash-yolo 系
                  （flash-yolo 复用 yolo26 损失；flash-yolo-pose 复用 ComputeLossPose）
    v3_*      ->  yolov3-tiny / yolov3-tiny-silu

本模块只做分派与公共导出；实现见同目录模块。接口同构：item_keys / __call__。
"""

from train.loss.v3_detect import ComputeLossV3
from train.loss.yolo26_detect import ComputeLoss
from train.loss.yolo26_pose import ComputeLossPose
from train.loss.yolo26_seg import ComputeLossSeg

__all__ = ["ComputeLoss", "ComputeLossSeg", "ComputeLossPose", "ComputeLossV3", "build_loss"]

_LOSS_BY_ARCH = {
    "yolo26": ComputeLoss,
    "flash-yolo": ComputeLoss,
    "yolo26-seg": ComputeLossSeg,
    "flash-yolo-seg": ComputeLossSeg,
    "yolo26-pose": ComputeLossPose,
    "flash-yolo-pose": ComputeLossPose,
    "yolov3-tiny": ComputeLossV3,
    "yolov3-tiny-silu": ComputeLossV3,
}


def build_loss(arch, cfg, head, device):
    """按架构分派损失实现（trainer 的单一入口）

    YOLOE-26 按头的状态二选一：PE 线性探测的 fused 头只训练 cv3 尾卷积，mask/proto
    分支全程冻结——掩码项没有可训练参数，走检测损失；未 fused 的文本布局（全量微调）
    沿用分割损失（检测 + 掩码 + semseg）。
    """
    if arch == "yoloe-26":
        return ComputeLoss(cfg, head, device) if getattr(head, "is_fused", False) \
            else ComputeLossSeg(cfg, head, device)
    cls = _LOSS_BY_ARCH.get(arch)
    if cls is None:
        raise ValueError(f"unknown arch {arch!r}; available: {sorted(set(_LOSS_BY_ARCH) | {'yoloe-26'})}")
    return cls(cfg, head, device)
