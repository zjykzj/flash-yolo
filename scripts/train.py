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
from utils.logger import get_logger, setup_logging
from utils.paths import increment_path

setup_logging()
logger = get_logger(__name__)

# TrainConfig 可调字段按类型分组（argparse 默认 None -> 不覆盖 yaml）
_INT_FIELDS = ["epochs", "batch", "nbs", "imgsz", "workers", "seed", "close_mosaic", "val_epochs", "val_limit",
               "limit", "ns_iters", "ema_tau", "topk", "topk_o2o", "topk2"]
_FLOAT_FIELDS = ["lr0", "lrf", "momentum", "weight_decay", "muon_w", "sgd_w", "warmup_epochs",
                 "box_gain", "cls_gain", "dfl_gain", "cls_w", "tal_alpha", "tal_beta",
                 "prog_alpha_init", "prog_alpha_final", "stal_s_min", "stal_s_ref", "ema_decay",
                 "mosaic", "mixup", "copy_paste", "aug_scale", "degrees", "shear", "translate",
                 "fliplr", "flipud", "hsv_h", "hsv_s", "hsv_v", "bgr"]


def main():
    parser = argparse.ArgumentParser(description="YOLO26 COCO training")
    parser.add_argument("--data", default=None, help="COCO data root (default: config/train.yaml data_dir)")
    parser.add_argument("--weights", default=None, help="init from .safetensors (finetune; default: from scratch)")
    parser.add_argument("--resume", default=None, help="resume.pt path or run dir (restores full training state)")
    parser.add_argument("--device", default=None, help="torch device (default: auto)")
    parser.add_argument("--name", default=None, help="run dir suffix (runs/train/train-<name>)")
    parser.add_argument("--scale", dest="scale", default=None, help="model scale (n/s/m/l/x)")
    parser.add_argument("--train-split", dest="train_split", default=None, help="training split (default train2017)")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None, help="mixed precision (default: on)")
    parser.add_argument("--channels-last", dest="channels_last", action=argparse.BooleanOptionalAction, default=None,
                        help="NHWC training memory format (default: yaml)")
    parser.add_argument("--cos-lr", dest="cos_lr", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--verbose", action="store_true", help="DEBUG logging (third-party output)")
    # TrainConfig 字段（默认 None -> 不覆盖 yaml）
    for f in _INT_FIELDS:
        parser.add_argument(f"--{f.replace('_', '-')}", dest=f, type=int, default=None, help=f"override train.yaml {f}")
    for f in _FLOAT_FIELDS:
        parser.add_argument(f"--{f.replace('_', '-')}", dest=f, type=float, default=None, help=f"override train.yaml {f}")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel("DEBUG")

    cfg = load_train_config(TRAIN_CONFIG_PATH)
    if args.data:
        cfg.data_dir = args.data
    apply_cli(cfg, args)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    run_dir = increment_path(ROOT / "runs" / "train" / ("train" + (f"-{args.name}" if args.name else "")))
    # 头部信息由 Trainer._print_startup 统一打印（避免重复）

    trainer = Trainer(cfg, device=device, run_dir=run_dir, resume=args.resume, weights=args.weights)
    trainer.train()


if __name__ == "__main__":
    main()
