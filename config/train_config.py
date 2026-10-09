"""训练配置：TrainConfig dataclass + yaml 加载（含配方合并）+ CLI 合并（与 config/train.yaml 同包相邻）

优先序：config/train.yaml 基础值 -> 配方文件覆盖（按 scale 取增量）-> CLI 显式字段
（argparse 默认 None，避免 yaml 被硬编码默认值遮蔽）。

配方解析（`--recipe`）：
- 缺省或 `default` = 内置基线（train.yaml 基础值，不做任何覆盖）
- 其他名字 -> `config/recipes/<name>.yaml`
- 含 .yaml/.yml 后缀 -> 按路径直接读取
配方文件结构：`base`（该配方的完整值）+ `scale_overrides.<scale>`（各档增量，n 档即 base）。
"""

import logging
from dataclasses import dataclass, fields
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

# 路径常量与超参分离：值在 yaml（本模块负责读/合并），此处只放路径
TRAIN_CONFIG_PATH = Path(__file__).resolve().parent / "train.yaml"
RECIPES_DIR = Path(__file__).resolve().parent / "recipes"


@dataclass
class TrainConfig:
    """训练超参（默认值 = 通用训练默认 recipe: default，见 config/train.yaml）"""

    recipe: str = "default"  # default = 内置基线（train.yaml）| 其他名字查 config/recipes/<name>.yaml

    # data / io
    data: str = ""  # 数据集描述符（config/datasets/[local/]<name>.yaml 或 .yaml 路径）；
                    # 空 = 未设置，必须由 CLI --data 提供（见 scripts/train.py）
    scale: str = "n"
    model: str = "yolo26"  # 架构名（model/build.py 的 ARCHS 注册表；yolo26 | yolov3-tiny | flash-yolo）
    cfg_path: str = ""  # 自定义模型 yaml 覆盖 ARCHS 注册路径（架构筛选实验用；"" = 用注册表）
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
    stop_after: int = 0  # >0 = 只跑前 N 轮（lr/close_mosaic 仍按 epochs 算——筛选实验用）

    # 训练产物（复盘/调优用，默认全开——关掉它们等于放弃事后分析能力）
    save_period: int = 20  # 每 N 轮存 weights/epochNNN.safetensors（保留最近 keep_periodic 个）
    keep_periodic: int = 3
    diag_interval: int = 50  # 每 N 个优化步写一行 diag/train_diag.csv（0 = 关闭）
    aug_samples: int = 8  # 首个 / close_mosaic 后 / 末轮各存一张增强抽样网格图（每张 N 个样本）

    # optimizer (MuSGD)
    lr0: float = 0.01
    lrf: float = 0.01
    momentum: float = 0.937
    weight_decay: float = 0.0005
    muon_w: float = 0.528
    sgd_w: float = 0.674
    warmup_epochs: float = 3.0
    warmup_momentum: float = 0.8  # warmup 起始动量（官方默认 0.8，线性爬升到 momentum）
    cos_lr: bool = False
    ns_iters: int = 5

    # loss
    box_gain: float = 7.5
    cls_gain: float = 0.5
    dfl_gain: float = 1.5
    obj_gain: float = 1.0  # v3 系 objectness 权重（yolo26 双头无独立 obj 项，不使用）
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
    copy_paste_mode: str = "off"  # off = 官方检测口径（无 segments 时 CopyPaste 恒为 no-op）| box = 矩形贴块近似
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


def resolve_recipe(name):
    """配方名 -> 文件路径

    - `default`（或缺省）= 内置基线（train.yaml 基础值，无覆盖），由调用方短路，不走本函数
    - 其他名字：查 `config/recipes/<name>.yaml`；含 .yaml/.yml 后缀则按路径直接读取
    """
    p = Path(name)
    if p.suffix in {".yaml", ".yml"}:
        if not p.is_file():
            raise FileNotFoundError(f"recipe file not found: {p}")
        return p
    candidate = RECIPES_DIR / f"{name}.yaml"
    if not candidate.is_file():
        available = sorted(x.stem for x in RECIPES_DIR.glob("*.yaml")) if RECIPES_DIR.is_dir() else []
        raise ValueError(f"unknown recipe {name!r}; available: {available} (or pass a .yaml path)")
    return candidate


def load_train_config(path, recipe=None, scale=None) -> TrainConfig:
    """yaml -> TrainConfig（优先序：基础值 -> 配方覆盖 -> 传入的 recipe/scale）

    Args:
        path: config/train.yaml 路径（基础值 + `recipe` 字段指定默认配方）
        recipe: 覆盖 yaml 的 recipe 字段（CLI --recipe）：`default` = 基础值，
                其他名字查 config/recipes/<name>.yaml，含后缀则按路径读取
        scale: 覆盖 yaml 的 scale 字段（配方按档位取增量）
    """
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    known = {fld.name for fld in fields(TrainConfig)}
    recipe = recipe or raw.get("recipe", "default")
    scale = scale or raw.get("scale", "n")

    overrides = {}
    if recipe != "default":
        rec_path = resolve_recipe(recipe)
        entry = yaml.safe_load(rec_path.read_text(encoding="utf-8")) or {}
        extra = set(entry) - {"base", "scale_overrides"}
        if extra:
            logger.warning(f"{rec_path.name} 未识别字段（忽略）: {sorted(extra)}")
        overrides = dict(entry.get("base") or {})
        per_scale = entry.get("scale_overrides") or {}
        if scale in per_scale:
            overrides.update(per_scale[scale])
        elif scale != "n":  # n 档即 base 本身
            raise ValueError(f"recipe {recipe!r} 没有 scale {scale!r}（提供: n, {sorted(per_scale)}）")
        logger.info(f"recipe: {recipe} -> {rec_path}")

    unknown = (set(raw) - known) | (set(overrides) - known)
    if unknown:
        logger.warning(f"未识别的配置字段（忽略）: {sorted(unknown)}")
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
