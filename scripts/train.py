"""YOLO26 COCO 训练（从零 / --weights 微调 / --resume 续训）

用法:
    python scripts/train.py --data /path/to/coco                                  # 从零全量
    python scripts/train.py --data /path/to/coco --limit 512 --epochs 3           # 真实数据冒烟
    python scripts/train.py --data /path/to/coco --weights x.safetensors          # 权重初始化微调
    python scripts/train.py --resume runs/train/trainN/resume.pt                  # 断点续训
"""

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # 仓库根目录入 sys.path

import torch

from config.defaults import TRAIN_CONFIG_PATH
from config.train import apply_cli, load_train_config
from train.trainer import Trainer
from utils.logger import attach_file_log, get_logger, setup_logging
from utils.paths import increment_path

# 控制台立刻可用；文件日志等 run 目录确定后挂（见 attach_file_log）
setup_logging(to_file=False)
logger = get_logger(__name__)

# TrainConfig 可调字段按类型分组（argparse 默认 None -> 不覆盖 yaml）
_INT_FIELDS = ["epochs", "batch", "nbs", "imgsz", "workers", "seed", "close_mosaic", "val_epochs", "val_limit",
               "limit", "ns_iters", "ema_tau", "topk", "topk_o2o", "topk2",
               "stop_after", "save_period", "keep_periodic", "diag_interval", "aug_samples"]
_FLOAT_FIELDS = ["lr0", "lrf", "momentum", "weight_decay", "muon_w", "sgd_w", "warmup_epochs", "warmup_momentum",
                 "box_gain", "cls_gain", "dfl_gain", "tal_alpha", "tal_beta",
                 "prog_alpha_init", "prog_alpha_final", "stal_s_min", "stal_s_ref", "ema_decay",
                 "mosaic", "mixup", "copy_paste", "aug_scale", "degrees", "shear", "translate",
                 "fliplr", "flipud", "hsv_h", "hsv_s", "hsv_v", "bgr"]
_STR_FIELDS = ["copy_paste_mode"]


def main():
    parser = argparse.ArgumentParser(description="YOLO26 COCO training")
    parser.add_argument("--data", default=None, help="COCO data root (required unless set in config/train.yaml)")
    parser.add_argument("--weights", default=None, help="init from .safetensors (finetune; default: from scratch)")
    parser.add_argument("--resume", default=None, help="resume.pt path or run dir (restores full training state)")
    parser.add_argument("--device", default=None, help="torch device (default: auto)")
    parser.add_argument("--name", default=None, help="run dir suffix (runs/train/train-<name>)")
    parser.add_argument("--scale", dest="scale", default=None, help="model scale (n/s/m/l/x)")
    parser.add_argument("--recipe", default=None, help="recipe name (config/recipes/<name>.yaml) or a .yaml path; "
                                                       "default = the built-in baseline (config/train.yaml values)")
    parser.add_argument("--train-split", dest="train_split", default=None, help="training split (default train2017)")
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
        cfg.data_dir = args.data
    if not cfg.data_dir:  # 配置里不写死机器路径，数据根目录走命令行
        parser.error("--data is required (config/train.yaml leaves data_dir empty by design)")
    apply_cli(cfg, args)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    if args.resume:
        # resume 复用 checkpoint 所在目录（Runner 也会按 ckpt["run_dir"] 覆盖）：不新建目录，
        # 日志也追加到同一个 run.log —— 一次训练只有一份完整日志
        resume_path = Path(args.resume)
        run_dir = resume_path if resume_path.is_dir() else resume_path.parent
    else:
        run_dir = increment_path(ROOT / "runs" / "train" / ("train" + (f"-{args.name}" if args.name else "")))
    attach_file_log(run_dir / "run.log")
    # 启动信息块由 Trainer 按构建时机分段打印（环境 -> 模型 -> 数据集 -> 组件 -> 起跑），见 train/trainer.py

    trainer = Trainer(cfg, device=device, run_dir=run_dir, resume=args.resume, weights=args.weights)
    trainer.train()


if __name__ == "__main__":
    main()
