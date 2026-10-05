"""训练配置：TrainConfig dataclass + yaml 加载 + CLI 合并（与 config/train.yaml 同包相邻）

优先序：config/train.yaml 为单一事实源，CLI 参数只覆盖显式提供的字段
（argparse 默认 None，避免 yaml 被硬编码默认值遮蔽）。
"""

import logging
import warnings
from dataclasses import dataclass, fields

import yaml

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    """训练超参（默认值 = 官方 yolo26n COCO 段配方，见 config/train.yaml）"""

    # data / io
    data_dir: str = "/home/zjykzj/datasets/coco"
    train_split: str = "train2017"  # 冒烟/调试可指到 val2017 子集
    scale: str = "n"
    epochs: int = 245
    batch: int = 16  # 物理 batch（开箱即用值；nbs 累积保证梯度语义不变）
    nbs: int = 64
    imgsz: int = 640
    workers: int = 8
    seed: int = 0
    close_mosaic: int = 10
    val_epochs: int = 1
    val_limit: int = 0
    limit: int = 0  # 训练子集（前 N 张；0 = 全量）

    # optimizer (MuSGD)
    lr0: float = 0.0054
    lrf: float = 0.0495
    momentum: float = 0.947
    weight_decay: float = 0.00064
    muon_w: float = 0.528
    sgd_w: float = 0.674
    warmup_epochs: float = 0.98
    cos_lr: bool = False
    ns_iters: int = 5

    # loss
    box_gain: float = 5.63
    cls_gain: float = 0.56
    dfl_gain: float = 9.04
    cls_w: float = 2.74
    tal_alpha: float = 0.5
    tal_beta: float = 6.0
    topk: int = 10
    topk_o2o: int = 7
    topk2: int = 1
    prog_alpha_init: float = 0.8
    prog_alpha_final: float = 0.1
    stal_s_min: float = 8.0
    stal_s_ref: float = 16.0

    # ema / amp
    ema_decay: float = 0.9999
    ema_tau: int = 2000
    amp: bool = True

    # augment
    mosaic: float = 0.909
    mixup: float = 0.012
    copy_paste: float = 0.075
    aug_scale: float = 0.562  # 仿射缩放增益（与模型档位 scale 字段区分）
    degrees: float = 1.11
    shear: float = 1.46
    translate: float = 0.071
    fliplr: float = 0.606
    flipud: float = 0.0
    hsv_h: float = 0.014
    hsv_s: float = 0.645
    hsv_v: float = 0.566
    bgr: float = 0.106


def load_train_config(path) -> TrainConfig:
    """yaml -> TrainConfig（未知 key 告警，不静默丢弃）"""
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    known = {fld.name for fld in fields(TrainConfig)}
    unknown = set(raw) - known
    if unknown:
        logger.warning(f"train.yaml 未识别字段（忽略）: {sorted(unknown)}")
    return TrainConfig(**{k: v for k, v in raw.items() if k in known})


def apply_cli(cfg: TrainConfig, args) -> TrainConfig:
    """CLI 覆盖：仅当参数显式提供（非 None）时覆盖对应字段"""
    for fld in fields(TrainConfig):
        val = getattr(args, fld.name, None)
        if val is not None:
            setattr(cfg, fld.name, val)
    return cfg
