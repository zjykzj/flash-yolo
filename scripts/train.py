"""检测 / 分割训练（YOLO26 / YOLO26-seg / YOLOv3-tiny；单卡或 DDP 多卡）

用法:
    python scripts/train.py --data /path/to/coco                                  # 从零全量（默认 yolo26）
    python scripts/train.py --data /path/to/coco --limit 512 --epochs 3           # 真实数据冒烟
    python scripts/train.py --data /path/to/coco --weights x.safetensors          # 权重初始化微调
    python scripts/train.py --data /path/to/coco --model yolov3-tiny              # 换架构
    python scripts/train.py --resume runs/train/trainN/resume.pt                  # 断点续训
    python scripts/train.py --data coco --device 0,1 --batch 64                   # 2 卡 DDP（batch 为全局值）
    torchrun --standalone --nproc_per_node=2 scripts/train.py --data coco ...     # torchrun 入口
"""

import argparse
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # 仓库根目录入 sys.path

import torch

from config.datasets import load_dataset
from config.train_config import TRAIN_CONFIG_PATH, apply_cli, load_train_config
from model.build import ARCHS, arch_task
from model.weights import arch_has_scale
from train.distributed import (broadcast_object, cleanup_distributed, free_port, init_distributed,
                               parse_device_ids)
from train.trainer import Trainer
from utils.logger import attach_file_log, configure_rank, get_logger, log_params, setup_logging
from utils.paths import increment_path

# 控制台立刻可用；文件日志等 run 目录确定后挂（见 attach_file_log）
setup_logging(to_file=False)
logger = get_logger(__name__)

# TrainConfig 可调字段按类型分组（argparse 默认 None -> 不覆盖 yaml）
_INT_FIELDS = ["epochs", "batch", "nbs", "imgsz", "workers", "seed", "close_mosaic", "val_epochs", "val_limit",
               "limit", "ns_iters", "ema_tau", "topk", "topk_o2o", "topk2",
               "stop_after", "save_period", "keep_periodic", "diag_interval", "aug_samples", "mask_ratio",
               "freeze"]
_FLOAT_FIELDS = ["lr0", "lrf", "momentum", "weight_decay", "muon_w", "sgd_w", "warmup_epochs", "warmup_momentum",
                 "warmup_bias_lr",
                 "lr_step_gamma", "box_gain", "cls_gain", "cls_w", "dfl_gain", "obj_gain", "tal_alpha", "tal_beta",
                 "prog_alpha_init", "prog_alpha_final", "stal_s_min", "stal_s_ref", "ema_decay",
                 "multi_scale", "mosaic", "mixup", "cutmix", "copy_paste", "aug_scale", "degrees", "shear",
                 "perspective", "translate",
                 "fliplr", "flipud", "hsv_h", "hsv_s", "hsv_v", "bgr"]
_STR_FIELDS = ["copy_paste_mode"]


def _build_parser():
    parser = argparse.ArgumentParser(description="YOLO26 / YOLO26-seg / YOLOv3-tiny training")
    parser.add_argument("--data", default=None,
                        help="dataset descriptor: a name in config/datasets/ (local/ wins) or a .yaml path")
    parser.add_argument("--weights", default=None, help="init from .safetensors (finetune; default: from scratch)")
    parser.add_argument("--resume", default=None, help="resume.pt path or run dir (restores full training state)")
    parser.add_argument("--device", default=None,
                        help="torch device: cuda:0 / cpu / N / 0,1 (DDP; training only). Default: auto")
    parser.add_argument("--name", default=None, help="run dir suffix (runs/train/train-<name>)")
    parser.add_argument("--scale", dest="scale", default=None,
                        help="model scale (yolo26 family: n/s/m/l/x; other architectures are fixed)")
    parser.add_argument("--model", dest="model", default=None, choices=sorted(ARCHS),
                        help="architecture (default: train.yaml `model`, yolo26)")
    parser.add_argument("--cfg", dest="cfg_path", default=None,
                        help="model yaml path override (custom structure; screening/experiment variants)")
    parser.add_argument("--recipe", default=None, help="recipe name (config/recipes/<name>.yaml) or a .yaml path; "
                                                       "default = the built-in baseline (config/train.yaml values)")
    parser.add_argument("--single-cls", dest="single_cls", action=argparse.BooleanOptionalAction, default=None,
                        help="single-class training: remap every label class to 0 and build an nc=1 head "
                             "(default: yaml `single_cls`, false)")
    parser.add_argument("--plots", action=argparse.BooleanOptionalAction, default=None,
                        help="write results/PR/confusion-matrix plots at the end of training "
                             "(default: yaml `plots`, true; needs matplotlib)")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None,
                        help="mixed precision fp16+GradScaler (default: yaml `amp`, false; from-scratch "
                             "training NaNs — intended for --weights finetune)")
    parser.add_argument("--channels-last", dest="channels_last", action=argparse.BooleanOptionalAction, default=None,
                        help="NHWC training memory format (default: yaml)")
    parser.add_argument("--cos-lr", dest="cos_lr", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=None,
                        help="bit-reproducible kernels: cudnn autotune off + deterministic algorithms "
                             "(default: yaml `deterministic`, true; measured cost is noise-level — "
                             "--no-deterministic trades reproducibility for speed)")
    parser.add_argument("--optimizer", default=None, choices=["musgd", "sgd"],
                        help="parameter update rule (default: yaml `optimizer`, musgd = Muon+SGD mix; "
                             "sgd = every parameter on the SGD path — darknet-style reproduction)")
    parser.add_argument("--nesterov", action=argparse.BooleanOptionalAction, default=None,
                        help="Nesterov correction on the SGD branch (default: yaml `nesterov`, true; "
                             "darknet's update is heavy-ball → use --no-nesterov to reproduce it)")
    parser.add_argument("--lr-schedule", dest="lr_schedule", default=None, choices=["linear", "cosine", "step"],
                        help="lr decay curve (default: yaml `lr_schedule`, linear; step = darknet-style drops)")
    parser.add_argument("--obj-iou-target", dest="obj_iou_target", action=argparse.BooleanOptionalAction, default=None,
                        help="v3 IoU-aware objectness: positive obj target = IoU(pred, matched GT) with 5x "
                             "positive weight (default: yaml `obj_iou_target`, false = darknet hard 0/1)")
    parser.add_argument("--overlap-mask", dest="overlap_mask", action=argparse.BooleanOptionalAction, default=None,
                        help="segmentation GT masks: one per-image instance index map (default: yaml `overlap_mask`, "
                             "true; --no-overlap-mask = one binary mask per instance)")
    parser.add_argument("--verbose", action="store_true", help="DEBUG logging (third-party output)")
    # TrainConfig 字段（默认 None -> 不覆盖 yaml）
    for f in _INT_FIELDS:
        parser.add_argument(f"--{f.replace('_', '-')}", dest=f, type=int, default=None, help=f"override train.yaml {f}")
    for f in _FLOAT_FIELDS:
        parser.add_argument(f"--{f.replace('_', '-')}", dest=f, type=float, default=None, help=f"override train.yaml {f}")
    for f in _STR_FIELDS:
        parser.add_argument(f"--{f.replace('_', '-')}", dest=f, choices=["off", "box", "flip"], default=None,
                            help=f"override train.yaml {f} (off = 官方检测口径 no-op | box = 矩形贴块近似 | "
                                 f"flip = 官方分割 CopyPaste)")
    return parser


def _check_spawn_devices(devices, parser):
    """父进程侧校验 `--device 0,1`（CUDA 可用 + 索引范围；子进程 init_distributed 还会再校验）"""
    if not torch.cuda.is_available():
        parser.error("--device lists multiple GPUs but CUDA is not available")
    count = torch.cuda.device_count()
    for dev in devices:
        if dev.type != "cuda" or dev.index is None:
            parser.error(f"invalid --device entry {dev}: DDP requires explicit CUDA indices (e.g. 0,1)")
        if dev.index >= count:
            parser.error(f"CUDA device index {dev.index} out of range (device_count={count})")


def _wait_children(procs):
    """等所有子进程退出；任一失败 -> 终止其余并返回非 0（torchrun 语义）"""
    code = 0
    try:
        while True:
            codes = [p.poll() for p in procs]
            bad = [c for c in codes if c is not None and c != 0]
            if bad:
                code = bad[0]
                break
            if all(c is not None for c in codes):
                return 0
            time.sleep(1.0)
    finally:
        for p in procs:
            if p.poll() is None:
                p.terminate()
    for p in procs:
        try:
            p.wait(timeout=30)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait()
    return code


def _maybe_spawn(args, parser):
    """`--device 0,1` 且当前不在 rendezvous 中 -> 每卡起一个子进程；返回退出码（非 DDP -> None）

    父进程只负责进程组环境变量与生命周期：RANK/LOCAL_RANK/WORLD_SIZE + 空闲 localhost 端口。
    子进程重新解析同一份 argv（看到 WORLD_SIZE 就不再 spawn），单卡路径完全不走这里。
    """
    if os.environ.get("WORLD_SIZE"):  # torchrun / 父进程已设好 rendezvous：本进程就是 worker
        return None
    try:
        devices = parse_device_ids(args.device)
    except ValueError as e:
        parser.error(str(e))
    if len(devices) <= 1:
        return None
    _check_spawn_devices(devices, parser)
    port = free_port()
    script = str(Path(__file__).resolve())
    procs = []
    try:
        for local_rank in range(len(devices)):
            env = os.environ.copy()
            env.update({
                "RANK": str(local_rank), "LOCAL_RANK": str(local_rank),
                "WORLD_SIZE": str(len(devices)),
                "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": str(port),
                "PYTHONUNBUFFERED": "1",
            })
            procs.append(subprocess.Popen([sys.executable, script, *sys.argv[1:]], env=env))
    except BaseException:  # Ctrl-C / 启动失败：别留下半组 worker
        for p in procs:
            p.terminate()
        raise
    return _wait_children(procs)


def main():
    parser = _build_parser()
    args = parser.parse_args()
    if args.verbose:
        logging.getLogger().setLevel("DEBUG")
    code = _maybe_spawn(args, parser)
    if code is not None:
        sys.exit(code)
    _run(args, parser)


def _run(args, parser):
    """单进程主体（DDP 时每个 rank 各跑一份；进程组在这里建立/销毁）"""
    try:
        info = init_distributed(args.device)
    except (ValueError, RuntimeError) as e:
        parser.error(str(e))
    configure_rank(info.rank)  # rank>0 控制台抬到 ERROR（文件日志仍全量）
    try:
        _train(args, parser, info)
    finally:
        cleanup_distributed()


def _train(args, parser, info):
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
    if arch_task(cfg.model) == "segment":
        if cfg.mask_ratio < 1 or cfg.imgsz % cfg.mask_ratio:
            parser.error(f"--mask-ratio must be >=1 and divide --imgsz (got {cfg.mask_ratio} / {cfg.imgsz})")
    if cfg.cfg_path and not Path(cfg.cfg_path).is_file():
        parser.error(f"--cfg not found: {cfg.cfg_path}")
    if info.enabled and cfg.batch % info.world_size:
        parser.error(f"--batch {cfg.batch} (global) must be divisible by world_size {info.world_size} "
                     f"for DDP (per-rank batch = {cfg.batch / info.world_size})")

    # 架构相关收尾：固定档位架构（v3 系）在此归一化供参数预览显示；Trainer 内有同款守卫
    if not arch_has_scale(cfg.model):
        if args.scale:
            logger.warning(f"--scale {args.scale} ignored for model {cfg.model!r} "
                           f"(scale is fixed to {ARCHS[cfg.model]['default_scale']!r})")
        cfg.scale = ARCHS[cfg.model]["default_scale"]
        # 配方与模型的匹配性告警在 load_train_config 里（配方自带 model 字段则视为面向该架构）

    if args.resume:
        # resume 复用 checkpoint 所在目录（Runner 也会按 ckpt["run_dir"] 覆盖）：不新建目录，
        # 日志也追加到同一个 run.log —— 一次训练只有一份完整日志
        resume_path = Path(args.resume)
        run_dir = resume_path if resume_path.is_dir() else resume_path.parent
    elif info.is_main:
        run_dir = increment_path(ROOT / "runs" / "train" / ("train" + (f"-{args.name}" if args.name else "")))
    else:
        run_dir = None
    # 只有 rank0 建目录，路径广播给其余 rank（避免 increment_path 的并发竞争）
    if info.enabled:
        run_dir = Path(broadcast_object(str(run_dir) if run_dir is not None else None, src=0))
    log_name = "run.log" if info.is_main else f"run.rank{info.rank}.log"
    attach_file_log(run_dir / log_name)
    log_params(logger, __file__, data=cfg.data, model=cfg.model, scale=cfg.scale, recipe=cfg.recipe,
               epochs=cfg.epochs, batch=cfg.batch, nbs=cfg.nbs, imgsz=cfg.imgsz,
               workers=cfg.workers, seed=cfg.seed,
               **(dict(mask_ratio=cfg.mask_ratio, overlap_mask=cfg.overlap_mask)
                  if arch_task(cfg.model) == "segment" else {}),
               # 单卡首行保持逐字符不变：DDP 才追加设备/规模
               **(dict(device=args.device, world_size=info.world_size) if info.enabled else {}))
    # 启动信息块由 Trainer 按构建时机分段打印（环境 -> 模型 -> 数据集 -> 组件 -> 起跑），见 train/trainer.py

    trainer = Trainer(cfg, device=info.device, run_dir=run_dir, resume=args.resume, weights=args.weights,
                      spec=spec, dist_info=info)
    trainer.train()


if __name__ == "__main__":
    main()
