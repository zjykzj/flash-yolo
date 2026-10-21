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
from dataclasses import dataclass, field, fields
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
    model: str = "yolo26"  # 架构名（model/build.py 的 ARCHS 注册表；yolo26 | yolo26-seg |
                          # yolov3-tiny | yolov3-tiny-silu（算子替换版）| flash-yolo）
    cfg_path: str = ""  # 自定义模型 yaml 覆盖 ARCHS 注册路径（架构筛选实验用；"" = 用注册表）
    epochs: int = 100
    batch: int = 16  # 物理 batch（开箱即用值；nbs 累积保证梯度语义不变）
    nbs: int = 64
    imgsz: int = 640
    multi_scale: float = 0.0  # 训练输入尺寸抖动（ultralytics 口径）：每 batch 在 imgsz×(1±r) 内按 32 量化随机；0 = 关
    channels_last: bool = True  # 训练走 NHWC（cuDNN 反向快 ~33%）；推理/导出链路不受影响
    workers: int = 8
    seed: int = 0
    deterministic: bool = True  # 逐位可复现：关 cudnn autotune + 确定性算法（代价实测在噪声内，
                                # 见 docs/train.md 复现口径）；关掉 = 纯速度优先
    close_mosaic: int = 10
    val_epochs: int = 1
    val_limit: int = 0
    limit: int = 0  # 训练子集（前 N 张；0 = 全量）
    stop_after: int = 0  # >0 = 只跑前 N 轮（lr/close_mosaic 仍按 epochs 算——筛选实验用）

    # segmentation（仅 task=segment 使用；检测路径不读）
    mask_ratio: int = 4  # GT 掩码下采样倍率：imgsz/mask_ratio；4 = 与 Proto26 同分辨率（官方默认）
    overlap_mask: bool = True  # True = 每图一张实例索引图（官方默认）；False = 每实例一张二值掩码

    # 训练产物（复盘/调优用，默认全开——关掉它们等于放弃事后分析能力）
    save_period: int = 20  # 每 N 轮存 weights/epochNNN.safetensors（保留最近 keep_periodic 个）
    plots: bool = True  # 训练结束画 results/PR/混淆矩阵（matplotlib 缺失时自动跳过）
    freeze: int = 0  # 冻结前 N 层（model.0..N-1；冻结层 BN 保持 eval）
    keep_periodic: int = 3
    diag_interval: int = 50  # 每 N 个优化步写一行 diag/train_diag.csv（0 = 关闭）
    aug_samples: int = 8  # 首个 / close_mosaic 后 / 末轮各存一张增强抽样网格图（每张 N 个样本）

    # optimizer (MuSGD)
    optimizer: str = "musgd"  # musgd = Muon + SGD 混合（官方 yolo26 口径）| sgd = 全参数走 SGD 路径
                              # （darknet 口径复刻：配 momentum 0.9 / nesterov false / wd 5e-4 / lr 0.001）
    nesterov: bool = True  # SGD 分支的 Nesterov 修正；darknet 的更新是 heavy-ball → 复刻口径下置 false
    lr0: float = 0.01
    lrf: float = 0.01
    momentum: float = 0.937
    weight_decay: float = 0.0005
    muon_w: float = 0.2  # released ultralytics default；官方 recipes 按 ckpt 记录覆盖（n 0.528）
    sgd_w: float = 1.0  # 同上
    warmup_epochs: float = 3.0
    warmup_momentum: float = 0.8  # warmup 起始动量（官方默认 0.8，线性爬升到 momentum）
    warmup_bias_lr: float = 0.0  # warmup 起始 bias lr（官方默认 0.1；0 = 历史行为，bias 与其它组同步从 0 爬升）
    cos_lr: bool = False
    lr_schedule: str = "linear"  # linear | cosine | step（step = darknet 台阶：lr_step_fracs 处各 ×gamma；
                                 # cosine 亦可用 cos_lr=true 触发；step 优先）
    lr_step_fracs: list = field(default_factory=lambda: [0.8, 0.9])  # step 的降调进度（epoch 进度比例）
    lr_step_gamma: float = 0.1  # step 每次降调的倍率
    ns_iters: int = 5

    # loss
    box_gain: float = 7.5
    cls_gain: float = 0.5
    cls_w: float = 1.0  # o2o 分类项额外权重（官方 ckpt 记录值；released 代码不消费，默认 1.0）
    dfl_gain: float = 1.5
    obj_gain: float = 1.0  # v3 系 objectness 权重（yolo26 双头无独立 obj 项，不使用）
    obj_iou_target: bool = False  # v3 系 IoU-aware objectness（参考 zjykzj/YOLOv3，--yolov3loss）：
                                  # 正样本 obj 目标 = IoU(预测框, 责任 GT)（常量）且权重 ×5；
                                  # false = darknet 硬目标 0/1（默认，行为不变）
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
    cutmix: float = 0.0
    copy_paste: float = 0.0
    copy_paste_mode: str = "off"  # off = 官方检测口径（无 segments 时 CopyPaste 恒为 no-op）| box = 矩形贴块近似
    aug_scale: float = 0.5  # 仿射缩放增益（与模型档位 scale 字段区分）
    degrees: float = 0.0
    shear: float = 0.0
    perspective: float = 0.0
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
    if merged.get("optimizer", "musgd") not in ("musgd", "sgd"):
        raise ValueError(f"unknown optimizer {merged['optimizer']!r} (musgd | sgd)")
    if recipe != "default" and "model" not in overrides:
        # 配方未声明 model 视为 yolo26 系配方（coco-ft/o365-pt 均如此）；自带 model 的配方
        # 面向特定架构，不告警。yolo26 系（含 seg / flash-yolo）复用同一套超参。
        from model.build import YOLO26_FAMILY  # 懒加载：config 模块不承担 torch 导入
        if merged.get("model", "yolo26") not in YOLO26_FAMILY:
            logger.warning(f"recipe {recipe!r} does not declare a model — it is a yolo26-tuned recipe; "
                           f"training {merged['model']!r} with it is not recommended")
    return TrainConfig(**merged)


def apply_cli(cfg: TrainConfig, args) -> TrainConfig:
    """CLI 覆盖：仅当参数显式提供（非 None）时覆盖对应字段"""
    for fld in fields(TrainConfig):
        val = getattr(args, fld.name, None)
        if val is not None:
            setattr(cfg, fld.name, val)
    return cfg
