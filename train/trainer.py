"""单卡训练主循环（AMP + MuSGD + EMA + ProgLoss + close_mosaic + 每 epoch 验证/保存）

结构上保持单设备语义（DDP 后置时只需加 sampler 与 rank0 守卫）。
"""

import csv
import logging
import sys
import time
from collections import deque
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import __version__
from config.train import TrainConfig
from data.coco import CocoDataset
from data.dataset import CocoTrainDataset, collate_fn, worker_init_fn
from model.summary import summary_lines
from model.weights import load_weights
from model.yolo26 import CONFIG_PATH, YOLO26
from train.checkpoint import load_resume, save_best_last, save_resume
from train.ema import ModelEMA
from train.loss import ComputeLoss
from train.lr import cosine_lr, linear_lr, set_epoch_lr, warmup_lr
from train.optimizer import MuSGD, build_param_groups
from train.validator import validate
from utils.logger import bold, redirect_prints
from utils.progress import ProgressBar

logger = logging.getLogger(__name__)


class Trainer:
    """YOLO26 训练器（--weights 初始化微调 / --resume 断点续训）"""

    def __init__(self, cfg: TrainConfig, device=None, run_dir=None, resume=None, weights=None):
        self.cfg = cfg
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.backends.cudnn.benchmark = True  # 训练不追求逐位可复现（resume 口径见 checkpoint）

        model = YOLO26(CONFIG_PATH, cfg.scale)
        if weights:
            load_weights(model, weights, strict=True)
            logger.info(f"initialized from {weights} (finetune)")
        model.to(self.device).train()
        self.model = model
        self.head = model.model[-1]

        self.optimizer = MuSGD(
            build_param_groups(model, cfg, self.head),
            lr=cfg.lr0,
            momentum=cfg.momentum,
            muon_w=cfg.muon_w,
            sgd_w=cfg.sgd_w,
            ns_iters=cfg.ns_iters,
        )
        self.ema = ModelEMA(model, decay=cfg.ema_decay, tau=cfg.ema_tau)
        # AMP（opt-in）：fp16 + GradScaler。从零训练默认关闭——半精度前向在 640² 会
        # NaN（BN eps=0.001 放大级联 + fp16 溢出 / bf16 尾数舍入，见 CLAUDE.md #10）
        self.scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp) if self.device.type == "cuda" else None
        self.loss_fn = ComputeLoss(cfg, self.head, self.device)
        t_data = time.monotonic()
        self.dataset = CocoTrainDataset(cfg, split=cfg.train_split, augment=True, limit=cfg.limit)
        self.data_load_time = time.monotonic() - t_data

        self.run_dir = Path(run_dir)
        self.best_fitness = -1.0  # 首次验证必存 best（mAP 为 0 时也落盘）
        self.start_epoch = 0
        if resume:
            path = Path(resume)
            if path.is_dir():
                path = path / "resume.pt"
            epoch, best, _, ckpt_run = load_resume(path, model, self.ema, self.optimizer, self.scaler)
            self.start_epoch = epoch + 1
            self.best_fitness = best
            if ckpt_run:
                self.run_dir = Path(ckpt_run)
            logger.info(f"resumed from {path} at epoch {self.start_epoch + 1} -> {self.run_dir}")

        self.results_csv = self.run_dir / "results.csv"
        self.accumulate = max(round(cfg.nbs / cfg.batch), 1)
        self.val_enabled = cfg.val_epochs > 0  # val_epochs=0 时全程不验证（小数据集冒烟用）
        self.val_dataset = None  # 懒构建（JSON 解析 ~数秒，仅需一次）
        self.csv_fields = ["epoch", "time_s", "box", "cls", "dfl", "o2m", "o2o", "loss", "lr", "mAP", "mAP50", "AR@100"]
        # close_mosaic 自动适配短跑：官方语义 = 最后 N 个 epoch 关 mosaic，但不超过总轮数的 1/5
        # （缩放结果在 _print_startup 的 mosaic 行体现）
        self.close_mosaic = min(cfg.close_mosaic, cfg.epochs // 5)

    # ---- 数据 ----
    def _dataloader(self, epoch):
        generator = torch.Generator().manual_seed(self.cfg.seed + epoch)  # 每 epoch 可复现抽样流
        return DataLoader(
            self.dataset,
            batch_size=self.cfg.batch,
            shuffle=True,
            num_workers=self.cfg.workers,
            collate_fn=collate_fn,
            worker_init_fn=worker_init_fn,
            generator=generator,
            pin_memory=self.device.type == "cuda",
        )

    # ---- 验证 ----
    def _validate(self):
        ema_model = self.ema.eval_model()
        if self.val_dataset is None:
            self.val_dataset = CocoDataset(self.cfg.data_dir, "val2017")
        # pycocotools 的裸 print 噪音 -> DEBUG（控制台只留一行 val 指标）
        with redirect_prints(logger):
            return validate(ema_model, self.val_dataset, self.device, limit=self.cfg.val_limit)

    # ---- 保存 ----
    def _save(self, epoch, is_best):
        save_best_last(self.run_dir, self.ema.ema, is_best)
        save_resume(
            self.run_dir / "resume.pt", self.model, self.ema, self.optimizer, self.scaler,
            epoch, self.best_fitness, self.cfg, self.run_dir,
        )

    # ---- 启动信息块 ----
    def _print_startup(self):
        """训练开始前打印完整参数快照（四模块：环境 / 超参 / 模型 / 数据，复盘无需翻 yaml）"""
        # ① 环境
        if self.device.type == "cuda":
            props = torch.cuda.get_device_properties(self.device)
            dev_str = f"CUDA:0 ({props.name}, {props.total_memory // 2**20}MiB)"
        else:
            dev_str = str(self.device)
        logger.info(bold(f"Flash-YOLO {__version__} 🚀 Python {sys.version.split()[0]} · torch {torch.__version__} · {dev_str}"))

        # 超参数快照（dict 直出，快速核对用）
        logger.info(f"hyperparameters: {asdict(self.cfg)}")

        # ② 模型：逐层参数表 + 汇总（profile_flops 会切 eval，打印后恢复 train）
        logger.info("")
        lines, n_layers, n_params, gflops = summary_lines(self.model, self.cfg.imgsz, device=self.device)
        for line in lines:
            logger.info(line)
        self.model.train()
        logger.info(bold(f"YOLO26{self.cfg.scale} summary: {n_layers} layers, {n_params:,} params, {gflops:.1f} GFLOPs"))

        # ③ 数据集
        logger.info("")
        logger.info(f"train: {self.cfg.train_split} {len(self.dataset)} images · {self.dataset.n_instances:,} instances · "
                    f"{self.dataset.n_categories} categories · load {self.data_load_time:.1f}s")
        if self.val_enabled:
            if self.val_dataset is None:
                self.val_dataset = CocoDataset(self.cfg.data_dir, "val2017")
            val_str = f"{len(self.val_dataset)} images · {self.val_dataset.n_instances:,} instances" + \
                (f" (limit {self.cfg.val_limit})" if self.cfg.val_limit else " (full)")
            logger.info(f"val:   val2017 {val_str}")
        else:
            logger.info("val:   disabled (val_epochs=0)")
        logger.info(f"image size {self.cfg.imgsz} · batch {self.cfg.batch} (nbs {self.cfg.nbs}, accum {self.accumulate}) "
                    f"· workers {self.cfg.workers} · AMP {'fp16' if self.cfg.amp else 'off'}")

        # ④ 训练组件与超参
        logger.info("")
        groups = self.optimizer.param_groups
        n_muon = sum(len(g["params"]) for g in groups if g.get("muon"))
        n_nodecay = sum(len(g["params"]) for g in groups if not g.get("muon") and g.get("wd", 0.0) == 0.0)
        n_lr3 = sum(len(g["params"]) for g in groups if g.get("lr_mult", 1.0) == 3.0)
        logger.info(f"optimizer: MuSGD(lr0={self.cfg.lr0}, momentum={self.cfg.momentum}, muon_w={self.cfg.muon_w}, "
                    f"sgd_w={self.cfg.sgd_w}) · groups: muon {n_muon} · no-decay {n_nodecay} · lr×3 {n_lr3}")
        close_epoch = max(self.cfg.epochs - self.close_mosaic, 0)
        logger.info(f"lr: warmup {self.cfg.warmup_epochs}ep -> {'cosine' if self.cfg.cos_lr else 'linear'} "
                    f"{self.cfg.lr0} -> {self.cfg.lr0 * self.cfg.lrf:.6f} · close_mosaic {self.close_mosaic} "
                    f"(scaled from {self.cfg.close_mosaic})")
        logger.info(f"loss: box {self.cfg.box_gain} CIoU · cls {self.cfg.cls_gain} BCE · l1 {self.cfg.dfl_gain} · "
                    f"cls_w {self.cfg.cls_w} · EMA {self.cfg.ema_decay} (tau {self.cfg.ema_tau})")
        logger.info(f"TAL topk {self.cfg.topk} / {self.cfg.topk_o2o}+{self.cfg.topk2} · STAL {self.cfg.stal_s_min}->"
                    f"{self.cfg.stal_s_ref}px · ProgLoss alpha {self.cfg.prog_alpha_init}->{self.cfg.prog_alpha_final} · "
                    f"mosaic {self.cfg.mosaic} (close from epoch {close_epoch})")
        logger.info(f"aug: copy_paste {self.cfg.copy_paste} · mixup {self.cfg.mixup} · fliplr {self.cfg.fliplr} · "
                    f"hsv {self.cfg.hsv_h}/{self.cfg.hsv_s}/{self.cfg.hsv_v} · seed {self.cfg.seed}")

        # 开始
        logger.info("")
        logger.info(f"results: {self.run_dir}")
        logger.info(bold(f"Starting training for {self.cfg.epochs} epochs..."))

    # ---- 主循环 ----
    def train(self):
        self._print_startup()

        for epoch in range(self.start_epoch, self.cfg.epochs):
            if epoch > self.start_epoch:
                logger.info("")  # epoch 间空行分隔（bar 不落日志，节奏靠它划分）
            # 指标表头（每轮重复；列统一 11 宽，与数据行对齐）
            logger.info("%11s" * 10 % ("Epoch", "GPU_mem", "box", "cls", "dfl", "o2m", "o2o", "Instances", "Size", "lr"))
            if epoch >= self.cfg.epochs - self.close_mosaic and self.cfg.mosaic > 0:
                self.dataset.close_mosaic()
                logger.info(f"close_mosaic: mosaic/mixup/copy_paste off from epoch {epoch + 1}")
            self.loss_fn.set_alpha(epoch, self.cfg.epochs)
            self.model.train()

            dl = self._dataloader(epoch)
            n_batch = len(dl)
            if n_batch == 0:
                raise RuntimeError("训练集为空（检查 data_dir 与 --limit）")
            bar = ProgressBar(n_batch, desc="")
            t0 = time.monotonic()
            t_window, n_window = t0, 0
            speed = None  # 10 batch 窗口速度（首个 batch 时立即计算）
            window = {k: deque(maxlen=10) for k in ("box", "cls", "dfl", "o2m", "o2o")}  # 近 10 batch 滑动均值
            sums = {"box": 0.0, "cls": 0.0, "dfl": 0.0, "o2m": 0.0, "o2o": 0.0, "total": 0.0}
            lr_last = 0.0

            for bi, (imgs, targets) in enumerate(dl):
                t = epoch + bi / max(n_batch - 1, 1)
                lr_base = warmup_lr(t, self.cfg.warmup_epochs) * (
                    cosine_lr(t, self.cfg.epochs, self.cfg.lr0, self.cfg.lrf)
                    if self.cfg.cos_lr
                    else linear_lr(t, self.cfg.epochs, self.cfg.lr0, self.cfg.lrf)
                )
                set_epoch_lr(self.optimizer, lr_base)
                lr_last = self.optimizer.param_groups[0]["lr"]

                imgs = imgs.to(self.device, non_blocking=True)
                targets = targets.to(self.device)
                with torch.autocast(device_type=self.device.type, enabled=self.cfg.amp and self.device.type == "cuda"):
                    preds = self.model(imgs)
                    loss, items = self.loss_fn(preds, targets, imgs.shape[0], self.cfg.imgsz)
                preds_bad = any(
                    not torch.isfinite(preds[b]["boxes"]).all() or not torch.isfinite(preds[b]["scores"]).all()
                    for b in preds
                )
                if not torch.isfinite(loss) or preds_bad:
                    # 快速诊断：定位首个非有限值所在分支 + 权重发散程度 + 损失分解
                    # 注意：权重 NaN 时 n_pos=0 会让损失保持有限（静默死亡），必须同时查 preds
                    wmax = max(p.abs().max().item() for p in self.model.parameters())
                    logger.error(f"non-finite loss/preds at epoch {epoch + 1} batch {bi} · items={items} · max|weight|={wmax:.2f}")
                    for branch in preds:
                        b, s = preds[branch]["boxes"], preds[branch]["scores"]
                        logger.error(
                            f"  {branch}: boxes finite={torch.isfinite(b).all().item()} max={b.abs().max().item():.2f} · "
                            f"scores finite={torch.isfinite(s).all().item()} max={s.abs().max().item():.2f}"
                        )
                    raise RuntimeError(f"non-finite loss/preds ({loss.item()}) — 见上方分支诊断")
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                if (bi + 1) % self.accumulate == 0 or bi == n_batch - 1:
                    if self.scaler is not None:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.ema.update(self.model)

                for k in sums:
                    if k == "total":
                        sums[k] += loss.detach().item() / self.cfg.batch
                    else:
                        sums[k] += items.get(k, 0.0)
                        window[k].append(items.get(k, 0.0))

                # 每 batch 刷新行内指标（近 10 batch 滑动均值 + lr + 显存）；
                # 速度按 10 batch 窗口计（相邻 batch 瞬时值噪声大）
                if (bi + 1) % 10 == 0 or bi == n_batch - 1 or speed is None:
                    now = time.monotonic()
                    speed = (bi + 1 - n_window) / max(now - t_window, 1e-6)
                    t_window, n_window = now, bi + 1
                mem = torch.cuda.memory_reserved() / 1e9 if self.device.type == "cuda" else 0.0
                mem_last = mem  # epoch 汇总行复用
                mean_w = {k: (sum(v) / len(v)) for k, v in window.items()}
                desc = (
                    "%11s" * 2 + "%11.3f" + "%11.4f" + "%11.3f" + "%11.2f" * 2 + "%11d" * 2 + "%11.5f"
                ) % (
                    f"{epoch + 1}/{self.cfg.epochs}", f"{mem:.2f}G",
                    mean_w["box"], mean_w["cls"], mean_w["dfl"], mean_w["o2m"], mean_w["o2o"],
                    len(dl.dataset), self.cfg.imgsz, lr_last,
                )
                bar.update(bi + 1, speed, desc=desc)
            bar.close()

            # ---- epoch 汇总（与表头同格式，落日志为最终记录）----
            mean = {k: v / n_batch for k, v in sums.items()}
            elapsed = time.monotonic() - t0
            epoch_line = (
                "%11s" * 2 + "%11.3f" + "%11.4f" + "%11.3f" + "%11.2f" * 2 + "%11d" * 2 + "%11.5f"
            ) % (
                f"{epoch + 1}/{self.cfg.epochs}", f"{mem_last:.2f}G",
                mean["box"], mean["cls"], mean["dfl"], mean["o2m"], mean["o2o"],
                len(dl.dataset), self.cfg.imgsz, lr_last,
            )
            logger.info(f"{epoch_line} · {elapsed:.1f}s · {n_batch / elapsed:.1f}it/s")

            # ---- 验证 ----
            metrics = None
            if self.val_enabled and ((epoch + 1) % self.cfg.val_epochs == 0 or epoch == self.cfg.epochs - 1):
                t_val = time.monotonic()
                metrics = self._validate()
                val_elapsed = time.monotonic() - t_val
                logger.info(
                    f"val: mAP@[.5:.95] {metrics['mAP@[.5:.95]']:.4f} · mAP@50 {metrics['mAP@50']:.4f} · "
                    f"AR {metrics['AR@100']:.4f} · {metrics['images']} images · {val_elapsed:.1f}s"
                )
                fitness = metrics["mAP@[.5:.95]"]
            else:
                fitness = self.best_fitness
            is_best = fitness > self.best_fitness
            if is_best:
                self.best_fitness = fitness
                logger.info(f"new best fitness {fitness:.4f} -> best.safetensors")

            self._save(epoch, is_best)
            self._append_results(epoch, elapsed, mean, lr_last, metrics)

        logger.info(f"training done -> {self.run_dir}")

    def _append_results(self, epoch, elapsed, mean, lr, metrics):
        row = {
            "epoch": epoch + 1, "time_s": round(elapsed, 1),
            "box": round(mean["box"], 4), "cls": round(mean["cls"], 4), "dfl": round(mean["dfl"], 4),
            "o2m": round(mean["o2m"], 4), "o2o": round(mean["o2o"], 4),
            "loss": round(mean["total"], 4), "lr": lr,
        }
        if metrics:
            row.update({"mAP": metrics["mAP@[.5:.95]"], "mAP50": metrics["mAP@50"], "AR@100": metrics["AR@100"]})
        new_file = not self.results_csv.exists()
        with open(self.results_csv, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=self.csv_fields)  # 固定列（无验证的 epoch 留空）
            if new_file:
                w.writeheader()
            w.writerow(row)
