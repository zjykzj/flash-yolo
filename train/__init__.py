"""训练模块（M3）：算法与训练循环（数据 I/O 与增强在 data/ 下）

每个文件独立可读、可单测：config(dataclass) / assigner / loss / optimizer /
ema / lr / checkpoint / validator / trainer。
"""

from train.assigner import TaskAlignedAssigner
from config.train import TrainConfig, apply_cli, load_train_config
from train.ema import ModelEMA
from train.loss import ComputeLoss
from train.optimizer import MuSGD, build_param_groups
from train.trainer import Trainer
from train.validator import validate

__all__ = [
    "TrainConfig",
    "load_train_config",
    "apply_cli",
    "TaskAlignedAssigner",
    "ComputeLoss",
    "MuSGD",
    "build_param_groups",
    "ModelEMA",
    "Trainer",
    "validate",
]
