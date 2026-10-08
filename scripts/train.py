"""检测训练（YOLO26 / YOLOv3-tiny；从零 / --weights 微调 / --resume 续训）

用法:
    python scripts/train.py --data /path/to/coco                                  # 从零全量（默认 yolo26）
    python scripts/train.py --data /path/to/coco --limit 512 --epochs 3           # 真实数据冒烟
    python scripts/train.py --data /path/to/coco --weights x.safetensors          # 权重初始化微调
    python scripts/train.py --data /path/to/coco --model yolov3-tiny              # 换架构
    python scripts/train.py --resume runs/train/trainN/resume.pt                  # 断点续训
"""

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # 仓库根目录入 sys.path

import torch

from config.datasets import load_dataset
from config.train_config import TRAIN_CONFIG_PATH, apply_cli, load_train_config
from model.build import ARCHS
from train.trainer import Trainer
from utils.logger import attach_file_log, get_logger, log_params, setup_logging
from utils.paths import increment_path

# 控制台立刻可用；文件日志等 run 目录确定后挂（见 attach_file_log）
setup_logging(to_file=False)
logger = get_logger(__name__)

# TrainConfig 可调字段按类型分组（argparse 默认 None -> 不覆盖 yaml）
_INT_FIELDS = ["epochs", "batch", "nbs", "imgsz", "workers", "seed", "close_mosaic", "val_epochs", "val_limit",
               "limit", "ns_iters", "ema_tau", "topk", "topk_o2o", "topk2",
               "stop_after", "save_period", "keep_periodic", "diag_interval", "aug_samples"]
_FLOAT_FIELDS = ["lr0", "lrf", "momentum", "weight_decay", "muon_w", "sgd_w", "warmup_epochs", "warmup_momentum",
                 "box_gain", "cls_gain", "dfl_gain", "obj_gain", "tal_alpha", "tal_beta",
                 "prog_alpha_init", "prog_alpha_final", "stal_s_min", "stal_s_ref", "ema_decay",
                 "mosaic", "mixup", "copy_paste", "aug_scale", "degrees", "shear", "translate",
                 "fliplr", "flipud", "hsv_h", "hsv_s", "hsv_v", "bgr"]
_STR_FIELDS = ["copy_paste_mode"]


def main():
    parser = argparse.ArgumentParser(description="YOLO26 / YOLOv3-tiny training")
    parser.add_argument("--data", default=None,
                        help="dataset descriptor: a name in config/datasets/ (local/ wins) or a .yaml path")
    parser.add_argument("--weights", default=None, help="init from .safetensors (finetune; default: from scratch)")
    parser.add_argument("--resume", default=None, help="resume.pt path or run dir (restores full training state)")
    parser.add_argument("--device", default=None, help="torch device (default: auto)")
    parser.add_argument("--name", default=None, help="run dir suffix (runs/train/train-<name>)")
    parser.add_argument("--scale", dest="scale", default=None, help="model scale (yolo26 only: n/s/m/l/x)")
    parser.add_argument("--model", dest="model", default=None, choices=sorted(ARCHS),
                        help="architecture (default: train.yaml `model`, yolo26)")
    parser.add_argument("--recipe", default=None, help="recipe name (config/recipes/<name>.yaml) or a .yaml path; "
                                                       "default = the built-in baseline (config/train.yaml values)")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None,
                        help="mixed precision fp16+GradScaler (default: yaml `amp`, false; from-scratch "
                             "training NaNs — intended for --weights finetune)")
    parser.add_argument("--channels-last", dest="channels_last", action=argparse.BooleanOptionalAction, default=None,
                        help="NHWC training memory format (default: yaml)")
    parser.add_argument("--cos-lr", dest="cos_lr", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--verbose", action="store_true", help="DEBUG logging (third-party output)")
    # TrainConfig 字段（默认 None -> 不覆盖 yaml）
    for f in _INT_FIELDS:
        parser.add_argument(f"--{f.replace('_', '-')}", dest=f, type=int, default=None, help=f"override train.yaml {f}")
    for f in _FLOAT_FIELDS:
        parser.add_argument(f"--{f.replace('_', '-')}", dest=f, type=float, default=None, help=f"override train.yaml {f}")
    for f in _STR_FIELDS:
        parser.add_argument(f"--{f.replace('_', '-')}", dest=f, choices=["off", "box"], default=None,
                            help=f"override train.yaml {f} (off = 官方检测口径 no-op | box = 矩形贴块近似)")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel("DEBUG")

    cfg = load_train_config(TRAIN_CONFIG_PATH, recipe=args.recipe, scale=args.scale)
    if args.data:
        cfg.data = args.data
    if not cfg.data:  # 数据集描述符必须显式提供（config/train.yaml 不写死任何数据集）
        parser.error("--data is required (a dataset descriptor: a name in config/datasets/ or a .yaml path)")
    try:
        spec = load_dataset(cfg.data)  # 描述符错误在建 run 目录之前拦下（无 traceback）
    except (ValueError, FileNotFoundError) as e:
        parser.error(str(e))
    apply_cli(cfg, args)

    # 架构相关收尾：v3 系档位由 yaml 固定（此处归一化供参数预览显示；Trainer 内有同款守卫）
    if cfg.model != "yolo26":
        if args.scale:
            logger.warning(f"--scale {args.scale} ignored for model {cfg.model!r} (scale is fixed to 'tiny')")
        cfg.scale = ARCHS[cfg.model]["default_scale"]
        if cfg.recipe != "default":
            logger.warning(f"recipe {cfg.recipe!r} is tuned for yolo26 — training {cfg.model} with it is not recommended")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    if args.resume:
        # resume 复用 checkpoint 所在目录（Runner 也会按 ckpt["run_dir"] 覆盖）：不新建目录，
        # 日志也追加到同一个 run.log —— 一次训练只有一份完整日志
        resume_path = Path(args.resume)
        run_dir = resume_path if resume_path.is_dir() else resume_path.parent
    else:
        run_dir = increment_path(ROOT / "runs" / "train" / ("train" + (f"-{args.name}" if args.name else "")))
    attach_file_log(run_dir / "run.log")
    log_params(logger, __file__, data=cfg.data, model=cfg.model, scale=cfg.scale, recipe=cfg.recipe,
               epochs=cfg.epochs, batch=cfg.batch, nbs=cfg.nbs, imgsz=cfg.imgsz,
               workers=cfg.workers, seed=cfg.seed)
    # 启动信息块由 Trainer 按构建时机分段打印（环境 -> 模型 -> 数据集 -> 组件 -> 起跑），见 train/trainer.py

    trainer = Trainer(cfg, device=device, run_dir=run_dir, resume=args.resume, weights=args.weights, spec=spec)
    trainer.train()


if __name__ == "__main__":
    main()
