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
from config.datasets import load_dataset
from config.train_config import TrainConfig
from data.build import build_eval_dataset, build_train_dataset
from data.loader import collate_fn, worker_init_fn
from data.scan import scan_summary
from model.summary import summary_lines
from model.weights import encode_anchors, load_weights
from model.build import ARCHS, arch_display_name, build_model
from train.checkpoint import load_resume, save_best_last, save_periodic, save_resume
from train.ema import ModelEMA
from train.loss import build_loss
from train.lr import cosine_lr, linear_lr, set_epoch_lr, warmup_lr, warmup_momentum
from train.optimizer import MuSGD, build_param_groups
from train.validator import VAL_HEADER, format_val_row, validate
from utils.logger import bold, log_file_only
from utils.paths import write_run_meta
from utils.progress import ProgressBar, fmt_elapsed, fmt_num
from utils.visualize import draw_target_grid

logger = logging.getLogger(__name__)


def _pred_tensors(preds):
    """训练口径 head 输出 -> [(名字, 张量)]：dict-of-dict（Detect 双分支）与 list（V3Detect 逐级）统一"""
    if isinstance(preds, dict):
        for branch in preds:
            yield f"{branch}.boxes", preds[branch]["boxes"]
            yield f"{branch}.scores", preds[branch]["scores"]
    else:
        for lvl, t in enumerate(preds):
            yield f"level{lvl}", t


class Trainer:
    """检测训练器（架构由 cfg.model 选择；--weights 初始化微调 / --resume 断点续训）"""

    # 损失项 -> (表头名, 定点精度)：逐字符兼容历史列（新损失项在此登记，未登记回退 k_loss/3 位）
    _LOSS_COLUMNS = {
        "box": ("box_loss", 3), "cls": ("cls_loss", 4), "l1": ("l1_loss", 3),
        "o2m": ("o2m_loss", 2), "o2o": ("o2o_loss", 2), "obj": ("obj_loss", 3),
    }

    def __init__(self, cfg: TrainConfig, device=None, run_dir=None, resume=None, weights=None, spec=None):
        self.cfg = cfg
        self.spec = spec or load_dataset(cfg.data)  # 描述符：nc 与角色路径的唯一来源
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.backends.cudnn.benchmark = True  # 训练不追求逐位可复现（resume 口径见 checkpoint）

        self.run_dir = Path(run_dir)
        self.best_fitness = -1.0  # 首次验证必存 best（mAP 为 0 时也落盘）
        self.start_epoch = 0
        self._n_corrupt_seen = 0  # 惰性 corrupt 累计（每 epoch 读一次数据集里的共享计数）

        # ① 环境 + 超参快照：在建模型、解析数据集之前打印（启动白屏只剩 import 时间）
        self._print_env()

        if cfg.model != "yolo26":  # 非 yolo26 架构的档位由 yaml 固定（yolov3-tiny = tiny）；--scale 不适用
            cfg.scale = ARCHS[cfg.model]["default_scale"]
        model = build_model(cfg.model, cfg.scale, cfg.imgsz, nc=self.spec.nc)
        if weights:
            load_weights(model, weights, strict=True)
        model.to(self.device).train()
        if cfg.channels_last:  # NHWC 训练（在 EMA deepcopy 之前，副本继承布局）
            model.to(memory_format=torch.channels_last)
        self.model = model
        self.head = model.model[-1]

        # 随权重走的模型参数（写入 best/last/epochN 的 safetensors header；读取侧可选消费，
        # 缺失不影响加载——见 model/weights.py::load_meta）
        self.weights_meta = {
            "arch": cfg.model, "scale": cfg.scale, "nc": self.spec.nc, "imgsz": cfg.imgsz,
            "names": [self.spec.names[i] for i in range(self.spec.nc)],
        }
        if hasattr(self.head, "anchors"):  # v3：固化模型实际使用的锚点（自定义重聚类后换 yaml 也能复现）
            self.weights_meta["anchors"] = encode_anchors(self.head.anchors)

        # ② 模型信息（逐层表 + 汇总；--weights 提示跟在模型段里）
        self._print_model_info(weights)

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
        self.loss_fn = build_loss(cfg.model, cfg, self.head, self.device)
        self.loss_keys = tuple(self.loss_fn.item_keys)  # 训练/验证/CSV 列名由此派生

        if resume:  # 只依赖 model/ema/optimizer/scaler，放在建数据集之前（输出顺序：模型 -> resume -> 数据）
            path = Path(resume)
            if path.is_dir():
                path = path / "resume.pt"
            epoch, best, ckpt_cfg, ckpt_run = load_resume(path, model, self.ema, self.optimizer, self.scaler)
            self.start_epoch = epoch + 1
            self.best_fitness = best
            if ckpt_run:
                self.run_dir = Path(ckpt_run)
            if ckpt_cfg.get("data") and ckpt_cfg["data"] != self.cfg.data:
                logger.warning(f"resume checkpoint was written with data={ckpt_cfg['data']!r}, now "
                               f"data={self.cfg.data!r} — the model is rebuilt from the current descriptor "
                               f"(class count may differ; strict weight load will catch a mismatch)")
            if ckpt_cfg.get("model", "yolo26") != self.cfg.model:
                logger.warning(f"resume checkpoint was written with model={ckpt_cfg.get('model', 'yolo26')!r}, "
                               f"now model={self.cfg.model!r} — the model is rebuilt from the current config")
            logger.info(f"resumed from {path} at epoch {self.start_epoch + 1} -> {self.run_dir}")

        self.accumulate = max(round(cfg.nbs / cfg.batch), 1)
        self.val_enabled = cfg.val_epochs > 0  # val_epochs=0 时全程不验证（小数据集冒烟用）
        self.val_dataset = None  # 懒构建（JSON 解析 ~数秒，仅需一次）
        self.csv_fields = ["epoch", "time_s", *self.loss_keys, "loss", "lr",
                           "mAP", "mAP50", "P", "R", "AR@100",
                           *(f"val_{k}" for k in self.loss_keys)]
        # close_mosaic 自动适配短跑：官方语义 = 最后 N 个 epoch 关 mosaic，但不超过总轮数的 1/5
        # （缩放结果在 _print_components 的 mosaic 行体现）
        self.close_mosaic = min(cfg.close_mosaic, cfg.epochs // 5)
        self.stop_after = cfg.stop_after if cfg.stop_after > 0 else cfg.epochs  # 筛选实验：跑到第 N 轮停

        # ③ 数据集（最慢的一段：解析 448MB ann json ~12s + 扫描 118k 图 ~2.6s，实时报进度）
        self._build_data()

        self.results_csv = self.run_dir / "results.csv"
        self.diag_path = self.run_dir / "diag" / "train_diag.csv"
        self.samples_dir = self.run_dir / "samples"
        self.diag_fields = ["epoch", "batch", "lr", "lr_x3", "gnorm", *self.loss_keys, "loss"]

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
            self.val_dataset = build_eval_dataset(self.spec, "val")
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
        row.update({k: round(items.get(k, 0.0), 5) for k in self.loss_keys})
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
            "data": {"descriptor": str(self.spec.source), "name": self.spec.name, "format": self.spec.format,
                     "path": str(self.spec.root) if self.spec.root is not None else None,
                     "images": len(self.dataset), "instances": self.dataset.n_instances,
                     "categories": self.dataset.n_categories,
                     "backgrounds": self.dataset.n_backgrounds, "missing": self.dataset.n_missing,
                     "crowd_excluded": self.dataset.n_crowd_excluded,
                     "parse_s": round(self.dataset.parse_time, 2), "scan_s": round(self.dataset.scan_time, 2)},
            "val": ({"images": len(self.val_dataset), "instances": self.val_dataset.n_instances,
                     "backgrounds": self.val_dataset.n_backgrounds, "missing": self.val_dataset.n_missing,
                     "parse_s": round(self.val_dataset.parse_time, 2), "scan_s": round(self.val_dataset.scan_time, 2)}
                    if self.val_dataset is not None else {"disabled": True}),
            "corrupt": self._n_corrupt_seen,  # 惰性统计：训练中被读出失败的图（启动时写入 0）
            "model": {"arch": self.cfg.model, "scale": self.cfg.scale,
                      "params": sum(p.numel() for p in self.model.parameters())},
            "train": {"accumulate": self.accumulate, "close_mosaic_from": max(self.cfg.epochs - self.close_mosaic, 0),
                      "stop_after": self.stop_after, "device": str(self.device)},
        })

    def _save(self, epoch, is_best):
        save_best_last(self.run_dir, self.ema.ema, is_best, raw_model=self.model, meta=self.weights_meta)
        if self.cfg.save_period and (epoch + 1) % self.cfg.save_period == 0:
            save_periodic(self.run_dir, self.ema.ema, epoch, keep=self.cfg.keep_periodic,
                          meta=self.weights_meta)
        save_resume(
            self.run_dir / "resume.pt", self.model, self.ema, self.optimizer, self.scaler,
            epoch, self.best_fitness, self.cfg, self.run_dir,
        )

    def _epoch_row(self, epoch, mem, means, n_img, lr):
        """训练数据行（与表头同构：11 宽右对齐）——进度条 desc 与 epoch 定格行共用，杜绝两处漂移

        损失列经 fmt_num：常规输出与定点格式逐字符一致，超宽（发散值）回退科学计数保列对齐。
        列由 loss_keys 派生（_LOSS_COLUMNS 固定精度，逐字符兼容历史列）。
        """
        row = "%11s" * 2 + "".join(
            fmt_num(means[k], 11, self._LOSS_COLUMNS.get(k, (k, 3))[1]) for k in self.loss_keys
        ) + "%11d" * 2 + "%11.5f"
        return row % (f"{epoch + 1}/{self.cfg.epochs}", f"{mem:.2f}G", n_img, self.cfg.imgsz, lr)

    # ---- 启动信息块（四段：① 环境 ② 模型 ③ 数据 ④ 组件；③ 在数据集构建时就地输出）----
    def _print_env(self):
        """① 环境 + 超参快照（最先打印：这两行只依赖 cfg/device，不依赖任何构建结果）"""
        if self.device.type == "cuda":
            props = torch.cuda.get_device_properties(self.device)
            dev_str = f"CUDA:0 ({props.name}, {props.total_memory // 2**20}MiB)"
        else:
            dev_str = str(self.device)
        logger.info(bold(f"Flash-YOLO {__version__} 🚀 Python {sys.version.split()[0]} · torch {torch.__version__} · {dev_str}"))

        # 超参数快照（dict 直出，快速核对用）
        logger.info(f"hyperparameters: {asdict(self.cfg)}")

    def _print_model_info(self, weights=None):
        """② 模型：逐层参数表 + 汇总（profile_flops 会切 eval，打印后恢复 train）

        汇总行由 summary_lines 统一输出（带档位名，如 YOLO26n），避免重复打印。
        """
        logger.info("")
        lines, _, _, _ = summary_lines(self.model, self.cfg.imgsz, device=self.device,
                                       name=arch_display_name(self.cfg.model, self.cfg.scale))
        for line in lines[:-1]:
            logger.info(line)
        logger.info(bold(lines[-1]))
        self.model.train()
        if weights:
            logger.info(f"initialized from {weights} (finetune)")

    def _build_data(self):
        """③ 数据集：扫描行（静态提示 + 进度条）由 data 层就地输出，这里补 `└` 汇总行与 run.log 记录

        本机实测 118k 图：解析 instances_train2017.json（448 MiB）≈12s（C 层阻塞、无法进度化，
        只打一行静态提示）+ 单趟扫描 ≈2.6s（进度条覆盖这一段）。
        """
        logger.info("")
        self.dataset = build_train_dataset(self.cfg, self.spec, "train", augment=True,
                                           limit=self.cfg.limit, progress=True)
        self._log_split("train", self.dataset)
        if self.val_enabled:
            self.val_dataset = build_eval_dataset(self.spec, "val", progress=True)
            self._log_split("val", self.val_dataset,
                            extra=f" · limit {self.cfg.val_limit}" if self.cfg.val_limit else "")
        else:
            logger.info("val:   disabled (val_epochs=0)")
        logger.info(f"imgsz {self.cfg.imgsz} · batch {self.cfg.batch} · nbs {self.cfg.nbs} (accum {self.accumulate}) · "
                    f"workers {self.cfg.workers} · seed {self.cfg.seed} · AMP {'fp16' if self.cfg.amp else 'off'}")

    def _log_split(self, label, ds, extra=""):
        """扫描行的续行（控制台 + run.log 都有）+ 往 run.log 补一份扫描计数

        进度条只走控制台（UI 元素不进 logger），所以文件日志里另落一行同等信息。
        """
        logger.info(scan_summary(ds, extra))
        log_file_only(f"{label}: {len(ds.images)} images · {ds.n_backgrounds} backgrounds · {ds.n_missing} missing",
                      name=logger.name)
        if ds.n_missing:
            logger.warning(f"{label}: {ds.n_missing} images missing on disk, excluded (e.g. {ds.first_missing})")

    def _print_components(self):
        """④ 训练组件与超参（optimizer / lr / loss / TAL / aug / 产物）"""
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
        if self.cfg.model == "yolo26":
            logger.info(f"loss: box {self.cfg.box_gain} CIoU · cls {self.cfg.cls_gain} BCE · l1 {self.cfg.dfl_gain} · "
                        f"EMA {self.cfg.ema_decay} (tau {self.cfg.ema_tau})")
            logger.info(f"TAL: topk o2m {self.cfg.topk} · o2o {self.cfg.topk_o2o}->{self.cfg.topk2} · "
                        f"STAL {self.cfg.stal_s_min}->{self.cfg.stal_s_ref}px · "
                        f"ProgLoss alpha {self.cfg.prog_alpha_init}->{self.cfg.prog_alpha_final}")
        else:
            logger.info(f"loss: box {self.cfg.box_gain} CIoU · obj {self.cfg.obj_gain} BCE · "
                        f"cls {self.cfg.cls_gain} BCE (anchor-matched) · "
                        f"EMA {self.cfg.ema_decay} (tau {self.cfg.ema_tau})")
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

    def _print_launch(self):
        """⑤ 收尾：run 元数据 + 结果目录 + 起跑行"""
        logger.info("")
        self._write_meta()  # run 元数据（环境/配置/数据规模/命令行）
        logger.info(f"results: {self.run_dir}")
        logger.info(bold(f"Starting training for {self.cfg.epochs} epochs..."))

    # ---- 主循环 ----
    def train(self):
        t_start = time.monotonic()  # 整段训练墙钟（含启动块与验证）
        self._print_components()  # ④ 组件（①环境/②模型/③数据 已在 __init__ 里按构建时机打印）
        self._print_launch()  # ⑤ 起跑

        for epoch in range(self.start_epoch, self.cfg.epochs):
            if epoch > self.start_epoch:
                logger.info("")  # epoch 间空行分隔（bar 不落日志，节奏靠它划分）
            # 指标表头（每轮重复；列统一 11 宽右对齐，与数据行同 6 格缩进；损失列由 loss_keys 派生）
            loss_headers = [self._LOSS_COLUMNS.get(k, (f"{k}_loss", 3))[0] for k in self.loss_keys]
            logger.info("      " + "%11s" * (2 + len(loss_headers) + 3) % (
                "Epoch", "GPU_peak", *loss_headers, "Instances", "Size", "lr"))
            if epoch >= self.cfg.epochs - self.close_mosaic and self.cfg.mosaic > 0:
                self.dataset.close_mosaic()
                logger.info(f"close_mosaic: mosaic/mixup/copy_paste off from epoch {epoch + 1}")
            if hasattr(self.loss_fn, "set_alpha"):  # ProgLoss 仅 yolo26 有
                self.loss_fn.set_alpha(epoch, self.cfg.epochs)
            self.model.train()
            self._opt_step = 0  # 诊断行按优化步计数（accum 边界对齐）

            dl = self._dataloader(epoch)
            n_batch = len(dl)
            if n_batch == 0:
                raise RuntimeError("training set is empty — check the dataset descriptor and --limit")
            bar = ProgressBar(n_batch, desc="")
            t0 = time.monotonic()
            speed = None  # 累计平均速度（tqdm 口径：n / 已耗时）
            sums = {**{k: 0.0 for k in self.loss_keys}, "total": 0.0}
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
                                     n=self.cfg.aug_samples, names=self.dataset.names)

                imgs = imgs.to(self.device, non_blocking=True,
                               memory_format=torch.channels_last if self.cfg.channels_last else torch.preserve_format)
                targets = targets.to(self.device)
                with torch.autocast(device_type=self.device.type, enabled=self.cfg.amp and self.device.type == "cuda"):
                    preds = self.model(imgs)
                    loss, items = self.loss_fn(preds, targets, imgs.shape[0], self.cfg.imgsz)
                # preds 诊断按 10 步窗口取样：每次 8 个 isfinite+bool 转换（同步）≈ 1.2ms，
                # NaN 检出延迟 ≤10 步（loss 本身的有限性检查仍每步执行）；两种 head 输出形态走同一枚举
                preds_bad = ((bi + 1) % 10 == 0 or bi == n_batch - 1) and any(
                    not torch.isfinite(t).all() for _, t in _pred_tensors(preds)
                )
                if not torch.isfinite(loss) or preds_bad:
                    # 快速诊断：定位首个非有限值所在分支 + 权重发散程度 + 损失分解
                    # 注意：权重 NaN 时 n_pos=0 会让损失保持有限（静默死亡），必须同时查 preds
                    wmax = max(p.abs().max().item() for p in self.model.parameters())
                    logger.error(f"non-finite loss/preds at epoch {epoch + 1} batch {bi} · items={items} · max|weight|={wmax:.2f}")
                    if isinstance(preds, dict):
                        for branch in preds:
                            b, s = preds[branch]["boxes"], preds[branch]["scores"]
                            logger.error(
                                f"  {branch}: boxes finite={torch.isfinite(b).all().item()} max={b.abs().max().item():.2f} · "
                                f"scores finite={torch.isfinite(s).all().item()} max={s.abs().max().item():.2f}"
                            )
                    else:
                        for name, t in _pred_tensors(preds):
                            logger.error(f"  {name}: finite={torch.isfinite(t).all().item()} "
                                         f"max={t.abs().max().item():.2f}")
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

            # 惰性 corrupt：worker 里读失败 +1（跨 fork 共享计数），每 epoch 最多一条告警
            n_bad = self.dataset.n_corrupt + (self.val_dataset.n_corrupt if self.val_dataset is not None else 0)
            if n_bad > self._n_corrupt_seen:
                logger.warning(f"unreadable images: +{n_bad - self._n_corrupt_seen} this epoch "
                               f"({n_bad} total) — replaced with blank samples")
                self._n_corrupt_seen = n_bad

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
        if self._n_corrupt_seen:  # 惰性 corrupt 汇总（启动时的 meta.json 里还是 0，重写一次补上）
            logger.warning(f"{self._n_corrupt_seen} unreadable images replaced by blank samples during the run")
            self._write_meta()
        if self.val_enabled:  # 正式口径评估提示（best.safetensors 仅在有验证时落盘）
            extra = f" --model {self.cfg.model}" if self.cfg.model != "yolo26" else ""
            logger.info(f"official eval: python scripts/eval.py --weights {self.run_dir / 'weights' / 'best.safetensors'} "
                        f"--data {self.spec.source}{extra}")

    def _append_results(self, epoch, elapsed, mean, lr, metrics):
        row = {"epoch": epoch + 1, "time_s": round(elapsed, 1)}
        row.update({k: round(mean[k], 4) for k in self.loss_keys})
        row.update({"loss": round(mean["total"], 4), "lr": lr})
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
