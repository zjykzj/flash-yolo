"""单卡训练主循环（AMP + MuSGD + EMA + ProgLoss + close_mosaic + 每 epoch 验证/保存）

结构上保持单设备语义（DDP 后置时只需加 sampler 与 rank0 守卫）。
"""

import csv
import logging
import sys
import time
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
from train.checkpoint import load_resume, save_best_last, save_periodic, save_resume
from train.ema import ModelEMA
from train.loss import ComputeLoss
from train.lr import cosine_lr, linear_lr, set_epoch_lr, warmup_lr, warmup_momentum
from train.optimizer import MuSGD, build_param_groups
from train.validator import VAL_HEADER, format_val_row, validate
from utils.logger import bold, log_file_only
from utils.paths import write_run_meta
from utils.progress import ProgressBar, fmt_elapsed, fmt_num
from utils.visualize import draw_target_grid

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
        if cfg.channels_last:  # NHWC 训练（在 EMA deepcopy 之前，副本继承布局）
            model.to(memory_format=torch.channels_last)
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
        self.csv_fields = ["epoch", "time_s", "box", "cls", "l1", "o2m", "o2o", "loss", "lr",
                           "mAP", "mAP50", "P", "R", "AR@100",
                           "val_box", "val_cls", "val_l1", "val_o2m", "val_o2o"]
        # close_mosaic 自动适配短跑：官方语义 = 最后 N 个 epoch 关 mosaic，但不超过总轮数的 1/5
        # （缩放结果在 _print_startup 的 mosaic 行体现）
        self.close_mosaic = min(cfg.close_mosaic, cfg.epochs // 5)
        self.stop_after = cfg.stop_after if cfg.stop_after > 0 else cfg.epochs  # 筛选实验：跑到第 N 轮停
        self.diag_path = self.run_dir / "diag" / "train_diag.csv"
        self.samples_dir = self.run_dir / "samples"
        self.diag_fields = ["epoch", "batch", "lr", "lr_x3", "gnorm", "box", "cls", "l1", "o2m", "o2o", "loss"]

    def _sample_epochs(self):
        """增强抽样可视化的轮次：首轮 / close_mosaic 生效首轮 / 末轮（去重）"""
        last = self.stop_after - 1
        return {self.start_epoch, self.cfg.epochs - self.close_mosaic, last} & set(range(self.start_epoch, self.stop_after))

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
        # loss_fn 同传：val 损失与训练损失同实现（复用同一次 backbone/neck 前向）
        return validate(ema_model, self.val_dataset, self.device, limit=self.cfg.val_limit,
                        loss_fn=self.loss_fn, imgsz=self.cfg.imgsz)

    # ---- 产物 ----
    def _write_diag(self, epoch, bi, gnorm, items, loss):
        """每 diag_interval 个优化步一行：梯度整体范数（clip 前）+ 分组 lr + 损失分解

        gnorm 是 clip_grad_norm_ 的返回值（裁剪前范数），唯一一次 device->host 同步；
        饱和度 = 10/gnorm，长期 <1 说明实际步长 ∝ lr（见 CLAUDE.md 训练诊断一节）。
        """
        import csv as _csv

        base = [g["lr"] for g in self.optimizer.param_groups if g.get("lr_mult", 1.0) == 1.0]
        row = {"epoch": epoch + 1, "batch": bi, "lr": min(base) if base else 0.0,
               "lr_x3": max((g["lr"] for g in self.optimizer.param_groups), default=0.0),
               "gnorm": round(gnorm, 3), "loss": round(loss, 5)}
        row.update({k: round(items.get(k, 0.0), 5) for k in ("box", "cls", "l1", "o2m", "o2o")})
        new_file = not self.diag_path.exists()
        self.diag_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.diag_path, "a", newline="", encoding="utf-8") as f:
            w = _csv.DictWriter(f, fieldnames=self.diag_fields)
            if new_file:
                w.writeheader()
            w.writerow(row)

    def _write_meta(self):
        write_run_meta(self.run_dir, {
            "config": asdict(self.cfg),
            "data": {"dir": self.cfg.data_dir, "train_split": self.cfg.train_split,
                     "images": len(self.dataset), "instances": self.dataset.n_instances,
                     "categories": self.dataset.n_categories},
            "model": {"scale": self.cfg.scale, "params": sum(p.numel() for p in self.model.parameters())},
            "train": {"accumulate": self.accumulate, "close_mosaic_from": max(self.cfg.epochs - self.close_mosaic, 0),
                      "stop_after": self.stop_after, "device": str(self.device)},
        })

    def _save(self, epoch, is_best):
        save_best_last(self.run_dir, self.ema.ema, is_best, raw_model=self.model)
        if self.cfg.save_period and (epoch + 1) % self.cfg.save_period == 0:
            save_periodic(self.run_dir, self.ema.ema, epoch, keep=self.cfg.keep_periodic)
        save_resume(
            self.run_dir / "resume.pt", self.model, self.ema, self.optimizer, self.scaler,
            epoch, self.best_fitness, self.cfg, self.run_dir,
        )

    def _epoch_row(self, epoch, mem, means, n_img, lr):
        """训练数据行（与表头同构：11 宽右对齐）——进度条 desc 与 epoch 定格行共用，杜绝两处漂移

        损失列经 fmt_num：常规输出与定点格式逐字符一致，超宽（发散值）回退科学计数保列对齐。
        """
        return (
            "%11s" * 2 + fmt_num(means["box"], 11, 3) + fmt_num(means["cls"], 11, 4)
            + fmt_num(means["l1"], 11, 3) + fmt_num(means["o2m"], 11, 2) + fmt_num(means["o2o"], 11, 2)
            + "%11d" * 2 + "%11.5f"
        ) % (
            f"{epoch + 1}/{self.cfg.epochs}", f"{mem:.2f}G", n_img, self.cfg.imgsz, lr,
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
        # 汇总行由 summary_lines 统一输出（带档位名，如 YOLO26n），避免重复打印
        logger.info("")
        lines, _, _, _ = summary_lines(self.model, self.cfg.imgsz, device=self.device, name=f"YOLO26{self.cfg.scale}")
        for line in lines[:-1]:
            logger.info(line)
        logger.info(bold(lines[-1]))
        self.model.train()

        # ③ 数据集
        logger.info("")
        logger.info(f"train: {self.cfg.train_split} @ {self.cfg.data_dir} · {len(self.dataset)} images · "
                    f"{self.dataset.n_instances:,} instances · {self.dataset.n_categories} categories · "
                    f"load {self.data_load_time:.1f}s")
        if self.val_enabled:
            if self.val_dataset is None:
                self.val_dataset = CocoDataset(self.cfg.data_dir, "val2017")
            val_str = f"{len(self.val_dataset)} images · {self.val_dataset.n_instances:,} instances" + \
                (f" (limit {self.cfg.val_limit})" if self.cfg.val_limit else " (full)")
            logger.info(f"val:   val2017 {val_str}")
        else:
            logger.info("val:   disabled (val_epochs=0)")
        logger.info(f"imgsz {self.cfg.imgsz} · batch {self.cfg.batch} · nbs {self.cfg.nbs} (accum {self.accumulate}) · "
                    f"workers {self.cfg.workers} · seed {self.cfg.seed} · AMP {'fp16' if self.cfg.amp else 'off'}")

        # ④ 训练组件与超参
        logger.info("")
        groups = self.optimizer.param_groups
        n_muon = sum(len(g["params"]) for g in groups if g.get("muon"))
        n_nodecay = sum(len(g["params"]) for g in groups if not g.get("muon") and g.get("wd", 0.0) == 0.0)
        n_lr3 = sum(len(g["params"]) for g in groups if g.get("lr_mult", 1.0) == 3.0)
        logger.info(f"optimizer: MuSGD(lr0 {self.cfg.lr0} · momentum {self.cfg.momentum} · wd {self.cfg.weight_decay} · "
                    f"muon_w {self.cfg.muon_w} · sgd_w {self.cfg.sgd_w} · ns_iters {self.cfg.ns_iters})")
        logger.info(f"           groups: muon {n_muon} · no-decay {n_nodecay} · lr×3 {n_lr3}")
        close_epoch = max(self.cfg.epochs - self.close_mosaic, 0)
        # 只有真被 min(close_mosaic, epochs//5) 缩过才提示缩放来源
        scale_note = f", scaled from {self.cfg.close_mosaic}" if self.close_mosaic != self.cfg.close_mosaic else ""
        logger.info(f"lr: warmup {self.cfg.warmup_epochs}ep -> {'cosine' if self.cfg.cos_lr else 'linear'} "
                    f"{self.cfg.lr0} -> {self.cfg.lr0 * self.cfg.lrf:.6f} · "
                    f"close_mosaic last {self.close_mosaic} epochs (from epoch {close_epoch}{scale_note})")
        logger.info(f"loss: box {self.cfg.box_gain} CIoU · cls {self.cfg.cls_gain} BCE · l1 {self.cfg.dfl_gain} · "
                    f"EMA {self.cfg.ema_decay} (tau {self.cfg.ema_tau})")
        logger.info(f"TAL: topk o2m {self.cfg.topk} · o2o {self.cfg.topk_o2o}->{self.cfg.topk2} · "
                    f"STAL {self.cfg.stal_s_min}->{self.cfg.stal_s_ref}px · "
                    f"ProgLoss alpha {self.cfg.prog_alpha_init}->{self.cfg.prog_alpha_final}")
        logger.info(f"aug: mosaic {self.cfg.mosaic} · copy_paste {self.cfg.copy_paste}({self.cfg.copy_paste_mode}) · "
                    f"mixup {self.cfg.mixup} · fliplr {self.cfg.fliplr} · flipud {self.cfg.flipud} · "
                    f"hsv h/s/v {self.cfg.hsv_h}/{self.cfg.hsv_s}/{self.cfg.hsv_v} · bgr {self.cfg.bgr}")
        if self.cfg.copy_paste > 0 and self.cfg.copy_paste_mode != "box":
            logger.info("      copy_paste ignored: 官方 CopyPaste 需要 instance segments，检测标签下恒为 no-op "
                        "（copy_paste_mode=box 可开启矩形贴块近似）")
        if self.cfg.save_period or self.cfg.diag_interval or self.cfg.aug_samples:
            logger.info(f"products: save_period {self.cfg.save_period or '-'} · "
                        f"diag_interval {self.cfg.diag_interval or '-'} · aug_samples {self.cfg.aug_samples or '-'}")
        logger.info(f"     affine (every sample): degrees +/-{self.cfg.degrees} · shear +/-{self.cfg.shear} · "
                    f"translate +/-{self.cfg.translate} · scale {1 - self.cfg.aug_scale:.3f}-"
                    f"{1 + self.cfg.aug_scale:.3f}")

        # 开始
        logger.info("")
        self._write_meta()  # run 元数据（环境/配置/数据规模/命令行）
        logger.info(f"results: {self.run_dir}")
        logger.info(bold(f"Starting training for {self.cfg.epochs} epochs..."))

    # ---- 主循环 ----
    def train(self):
        t_start = time.monotonic()  # 整段训练墙钟（含启动块与验证）
        self._print_startup()

        for epoch in range(self.start_epoch, self.cfg.epochs):
            if epoch > self.start_epoch:
                logger.info("")  # epoch 间空行分隔（bar 不落日志，节奏靠它划分）
            # 指标表头（每轮重复；列统一 11 宽右对齐，与数据行同 6 格缩进）
            logger.info("      " + "%11s" * 10 % ("Epoch", "GPU_peak", "box_loss", "cls_loss", "l1_loss",
                                                "o2m_loss", "o2o_loss", "Instances", "Size", "lr"))
            if epoch >= self.cfg.epochs - self.close_mosaic and self.cfg.mosaic > 0:
                self.dataset.close_mosaic()
                logger.info(f"close_mosaic: mosaic/mixup/copy_paste off from epoch {epoch + 1}")
            self.loss_fn.set_alpha(epoch, self.cfg.epochs)
            self.model.train()
            self._opt_step = 0  # 诊断行按优化步计数（accum 边界对齐）

            dl = self._dataloader(epoch)
            n_batch = len(dl)
            if n_batch == 0:
                raise RuntimeError("训练集为空（检查 data_dir 与 --limit）")
            bar = ProgressBar(n_batch, desc="")
            t0 = time.monotonic()
            speed = None  # 累计平均速度（tqdm 口径：n / 已耗时）
            sums = {"box": 0.0, "cls": 0.0, "l1": 0.0, "o2m": 0.0, "o2o": 0.0, "total": 0.0}
            lr_last = 0.0

            for bi, (imgs, targets) in enumerate(dl):
                t = epoch + bi / max(n_batch - 1, 1)
                lr_base = warmup_lr(t, self.cfg.warmup_epochs) * (
                    cosine_lr(t, self.cfg.epochs, self.cfg.lr0, self.cfg.lrf)
                    if self.cfg.cos_lr
                    else linear_lr(t, self.cfg.epochs, self.cfg.lr0, self.cfg.lrf)
                )
                # warmup 期间动量线性爬升（官方训练循环同口径；之后恒为 cfg.momentum）
                mom = warmup_momentum(t, self.cfg.warmup_epochs, self.cfg.momentum, self.cfg.warmup_momentum)
                set_epoch_lr(self.optimizer, lr_base, momentum=mom)
                lr_last = self.optimizer.param_groups[0]["lr"]

                # 增强抽样可视化：首个 / close_mosaic 后首个 / 末轮各存一张网格图
                if bi == 0 and self.cfg.aug_samples and epoch in self._sample_epochs():
                    draw_target_grid(imgs, targets, self.samples_dir / f"epoch{epoch + 1:03d}.png",
                                     n=self.cfg.aug_samples)

                imgs = imgs.to(self.device, non_blocking=True,
                               memory_format=torch.channels_last if self.cfg.channels_last else torch.preserve_format)
                targets = targets.to(self.device)
                with torch.autocast(device_type=self.device.type, enabled=self.cfg.amp and self.device.type == "cuda"):
                    preds = self.model(imgs)
                    loss, items = self.loss_fn(preds, targets, imgs.shape[0], self.cfg.imgsz)
                # preds 诊断按 10 步窗口取样：每次 8 个 isfinite+bool 转换（同步）≈ 1.2ms，
                # NaN 检出延迟 ≤10 步（loss 本身的有限性检查仍每步执行）
                preds_bad = ((bi + 1) % 10 == 0 or bi == n_batch - 1) and any(
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
                    # 梯度全局范数裁剪 max_norm=10（官方 optimizer_step 同口径，无条件执行）：
                    # 缺它时 o2o 分类头在"小数据 + 快速到高 lr"下会进入 BN 掩蔽式复利增长
                    # （train loss 正常但 one2one_cv3 权重 0.68→5.8e18，5-7 轮 NaN；官方同配
                    # 方同数据 12 轮稳定）。AMP 路径须先 unscale 再裁剪（官方同顺序）。
                    if self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)
                        gnorm_t = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        gnorm_t = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                        self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.ema.update(self.model)
                    # 诊断行：裁剪前梯度范数（clip_grad_norm_ 返回值，免额外反向/同步）
                    self._opt_step += 1
                    if self.cfg.diag_interval and self._opt_step % self.cfg.diag_interval == 0:
                        self._write_diag(epoch, bi, float(gnorm_t), items, loss.detach().item())

                for k in sums:
                    if k == "total":
                        sums[k] += loss.detach().item() / self.cfg.batch
                    else:
                        sums[k] += items.get(k, 0.0)

                # 每 10 batch 刷新行内指标（ultralytics tloss 口径）：
                #   损失 = epoch 内累计运行均值（行内值在 epoch 末尾自然收敛到定格值，无断层）
                #   速度 = 累计平均（tqdm 口径，不再 10 batch 窗口跳动）
                #   显存 = 进程峰值（max_memory_reserved 单调、不跳动，判 OOM 风险才有意义）
                if (bi + 1) % 10 == 0 or bi == n_batch - 1 or speed is None:
                    speed = (bi + 1) / max(time.monotonic() - t0, 1e-6)
                mem = torch.cuda.max_memory_reserved() / 1e9 if self.device.type == "cuda" else 0.0
                mem_last = mem  # epoch 汇总行复用
                mean_w = {k: v / (bi + 1) for k, v in sums.items()}
                desc = "      " + self._epoch_row(epoch, mem, mean_w, len(dl.dataset), lr_last)
                bar.update(bi + 1, speed, desc=desc)
            # ---- epoch 定格（ultralytics 机制）：进度条行即记录，结束后保留 ----
            mean = {k: v / n_batch for k, v in sums.items()}
            elapsed = time.monotonic() - t0
            row = "      " + self._epoch_row(epoch, mem_last, mean, len(dl.dataset), lr_last)
            bar.update(n_batch, speed, desc=row)  # 末次刷新换真均值（非 10 batch 窗口值）
            bar.close()  # 换行保留，不再清屏
            # 控制台定格行=进度条行；文件日志另落真均值行（time_s 同时入 results.csv）
            log_file_only(row + f" · {fmt_elapsed(elapsed)}", name=logger.name)

            # ---- 验证（训练中趋势口径；正式数字用 scripts/eval.py）----
            metrics = None
            if self.val_enabled and ((epoch + 1) % self.cfg.val_epochs == 0 or epoch == self.cfg.epochs - 1):
                t_val = time.monotonic()
                logger.info(VAL_HEADER)  # 表头先行（进度条实时行挂在表头下）；AR@100 不展示，保留在 results.csv
                metrics = self._validate()
                val_elapsed = time.monotonic() - t_val
                # 控制台定格行=进度条行；文件日志另落一行（带评估耗时）
                log_file_only(format_val_row(metrics) + f" · {fmt_elapsed(val_elapsed)}", name=logger.name)
                fitness = metrics["mAP@[.5:.95]"]
            else:
                fitness = self.best_fitness
            is_best = fitness > self.best_fitness
            if is_best:
                self.best_fitness = fitness
                logger.info(f"new best fitness {fitness:.4f} -> best.safetensors")

            self._save(epoch, is_best)
            self._append_results(epoch, elapsed, mean, lr_last, metrics)
            if self.cfg.stop_after and epoch + 1 >= self.stop_after:  # 筛选实验：调度按 epochs 走
                logger.info(f"stop_after={self.stop_after} reached — 提前结束（lr/close_mosaic 仍按 "
                            f"{self.cfg.epochs} 轮调度，便于与全长 run 同轮次对比）")
                break

        elapsed_total = time.monotonic() - t_start
        n_epochs = min(self.stop_after, self.cfg.epochs) - self.start_epoch
        logger.info(f"training done -> {self.run_dir} · {n_epochs} epoch{'s' if n_epochs != 1 else ''} "
                    f"completed in {fmt_elapsed(elapsed_total)}")
        if self.val_enabled:  # 正式口径评估提示（best.safetensors 仅在有验证时落盘）
            logger.info(f"official eval: python scripts/eval.py --weights {self.run_dir / 'weights' / 'best.safetensors'} "
                        f"--data {self.cfg.data_dir}")

    def _append_results(self, epoch, elapsed, mean, lr, metrics):
        row = {
            "epoch": epoch + 1, "time_s": round(elapsed, 1),
            "box": round(mean["box"], 4), "cls": round(mean["cls"], 4), "l1": round(mean["l1"], 4),
            "o2m": round(mean["o2m"], 4), "o2o": round(mean["o2o"], 4),
            "loss": round(mean["total"], 4), "lr": lr,
        }
        if metrics:
            row.update({"mAP": metrics["mAP@[.5:.95]"], "mAP50": metrics["mAP@50"],
                        "P": metrics["P"], "R": metrics["R"], "AR@100": metrics["AR@100"]})
            if metrics.get("loss"):
                row.update({f"val_{k}": round(v, 4) for k, v in metrics["loss"].items()})
        new_file = not self.results_csv.exists()
        with open(self.results_csv, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=self.csv_fields)  # 固定列（无验证的 epoch 留空）
            if new_file:
                w.writeheader()
            w.writerow(row)
