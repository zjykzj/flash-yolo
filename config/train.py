"""训练配置：TrainConfig dataclass + yaml 加载（含配方合并）+ CLI 合并（与 config/train.yaml 同包相邻）

优先序：config/train.yaml 基础值 -> `recipes.<recipe>` 覆盖（official 按 scale 取增量）
-> CLI 显式字段（argparse 默认 None，避免 yaml 被硬编码默认值遮蔽）。
"""

import logging
from dataclasses import dataclass, fields

import yaml

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    """训练超参（默认值 = 通用训练默认 recipe: default，见 config/train.yaml）"""

    recipe: str = "default"  # default = 通用默认 | official = 官方 YOLO26 发布配方（按 scale 分档）

    # data / io
    data_dir: str = "/home/zjykzj/datasets/coco"
    train_split: str = "train2017"  # 冒烟/调试可指到 val2017 子集
    scale: str = "n"
    epochs: int = 100
    batch: int = 16  # 物理 batch（开箱即用值；nbs 累积保证梯度语义不变）
    nbs: int = 64
    imgsz: int = 640
    channels_last: bool = True  # 训练走 NHWC（cuDNN 反向快 ~33%）；推理/导出链路不受影响
    workers: int = 8
    seed: int = 0
    close_mosaic: int = 10
    val_epochs: int = 1
    val_limit: int = 0
    limit: int = 0  # 训练子集（前 N 张；0 = 全量）

    # optimizer (MuSGD)
    lr0: float = 0.01
    lrf: float = 0.01
    momentum: float = 0.937
    weight_decay: float = 0.0005
    muon_w: float = 0.528
    sgd_w: float = 0.674
    warmup_epochs: float = 3.0
    cos_lr: bool = False
    ns_iters: int = 5

    # loss
    box_gain: float = 7.5
    cls_gain: float = 0.5
    dfl_gain: float = 1.5
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
    amp: bool = False

    # augment
    mosaic: float = 1.0
    mixup: float = 0.0
    copy_paste: float = 0.0
    aug_scale: float = 0.5  # 仿射缩放增益（与模型档位 scale 字段区分）
    degrees: float = 0.0
    shear: float = 0.0
    translate: float = 0.1
    fliplr: float = 0.5
    flipud: float = 0.0
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    bgr: float = 0.0


def load_train_config(path, recipe=None, scale=None) -> TrainConfig:
    """yaml -> TrainConfig（优先序：基础值 -> recipe 覆盖 -> 传入的 recipe/scale）

    Args:
        path: config/train.yaml 路径
        recipe: 覆盖 yaml 的 recipe 字段（CLI --recipe）
        scale: 覆盖 yaml 的 scale 字段（official 配方按档位取增量）
    """
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    recipes = raw.get("recipes") or {}
    known = {fld.name for fld in fields(TrainConfig)}
    recipe = recipe or raw.get("recipe", "default")
    scale = scale or raw.get("scale", "n")

    overrides = {}
    if recipe != "default":
        entry = recipes.get(recipe)
        if entry is None:
            raise ValueError(f"unknown recipe {recipe!r} (train.yaml 提供: {sorted(recipes)})")
        overrides = dict(entry.get("base") or {})
        per_scale = entry.get("scale_overrides") or {}
        if scale in per_scale:
            overrides.update(per_scale[scale])
        elif scale != "n":  # n 档即 base 本身
            raise ValueError(f"recipe {recipe!r} 没有 scale {scale!r} 的配方（提供: n, {sorted(per_scale)}）")

    unknown = (set(raw) - known - {"recipes"}) | (set(overrides) - known)
    if unknown:
        logger.warning(f"train.yaml 未识别字段（忽略）: {sorted(unknown)}")
    merged = {k: v for k, v in raw.items() if k in known}
    merged.update({k: v for k, v in overrides.items() if k in known})
    merged["recipe"] = recipe
    merged["scale"] = scale
    return TrainConfig(**merged)


def apply_cli(cfg: TrainConfig, args) -> TrainConfig:
    """CLI 覆盖：仅当参数显式提供（非 None）时覆盖对应字段"""
    for fld in fields(TrainConfig):
        val = getattr(args, fld.name, None)
        if val is not None:
            setattr(cfg, fld.name, val)
    return cfg
