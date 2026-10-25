"""DDP 冒烟 worker（由 tests/test_distributed.py 以子进程启动，pytest 不收集本文件）

每个进程：init_distributed -> rank0 建 run 目录并广播 -> Trainer 训练 -> 写参数 SHA-256。
只哈希 `named_parameters()`：BN buffers / EMA 副本允许按 rank 不同（DDP 默认 buffer 同步
以 rank0 为准），参数经 all-reduce 后必须逐位一致。
"""

import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import torch

from config.train_config import TrainConfig
from train.distributed import broadcast_object, cleanup_distributed, init_distributed
from train.trainer import Trainer
from utils.logger import attach_file_log, configure_rank, setup_logging


def param_hash(model):
    """参数（不含 buffers）的稳定 SHA-256：名字 + 原始字节，按名字排序"""
    digest = hashlib.sha256()
    params = dict(model.named_parameters())
    for name in sorted(params):
        digest.update(name.encode())
        digest.update(params[name].detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="DDP training worker (test fixture)")
    parser.add_argument("--data", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--model", default="yolo26")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--nbs", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=160)
    parser.add_argument("--stop-after", type=int, default=0)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--out", required=True, help="parameter SHA-256 output file")
    args = parser.parse_args()

    setup_logging(to_file=False)
    info = init_distributed("cpu")  # env:// rendezvous from the parent test
    configure_rank(info.rank)
    try:
        run_dir = None
        if info.is_main:  # 与 scripts/train.py 同款：rank0 决定目录，其余 rank 收广播
            run_dir = str(Path(args.run_dir).resolve())
            Path(run_dir).mkdir(parents=True, exist_ok=True)
        run_dir = broadcast_object(run_dir, src=0)
        # 与 scripts/train.py 同款日志落盘：rank0 -> run.log，其余 -> run.rankN.log
        attach_file_log(Path(run_dir) / ("run.log" if info.is_main else f"run.rank{info.rank}.log"))
        cfg = TrainConfig(
            data=args.data, model=args.model, epochs=args.epochs, batch=args.batch, nbs=args.nbs,
            imgsz=args.imgsz, workers=0, val_epochs=1, val_limit=4, limit=0, amp=False,
            mask_ratio=4, overlap_mask=True, mosaic=0.0, mixup=0.0, cutmix=0.0, copy_paste=0.0,
            close_mosaic=0, warmup_epochs=0.0, aug_samples=0, plots=False, save_period=0,
            diag_interval=0, stop_after=args.stop_after,
        )
        trainer = Trainer(cfg, device=info.device, run_dir=run_dir, resume=args.resume, dist_info=info)
        trainer.train()
        Path(args.out).write_text(param_hash(trainer.model))
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
