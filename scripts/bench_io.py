"""训练 I/O 吞吐基准：为当前服务器挑 workers / threads / batch 组合

两阶段自动收敛（避免全网格爆炸）：

  ① loader-only 全网格（workers × threads，纯 CPU，无 GPU）→ 取吞吐前 K 组
  ② end-to-end 只跑 (batch × 前 K 组) 的完整训练步（fwd + loss + assign + bwd +
     clip + opt + EMA）→ 真实吞吐排名 + 推荐组合

为什么两阶段都必要：**加载器快 ≠ 训练快**。2026-10 实测（RTX 5090 / 25 核 / batch 64）：
loader-only 可达 496 img/s，但端到端只有 ~230 img/s，且把 workers 从 16 提到 20 后
端到端反而更慢（20 个单线程 worker 把 25 核占满，主进程的 collate/H2D/kernel 发射被饿）。
只有端到端数字能决策。

worker 线程数必须显式设置：DataLoader 是多进程，每个 worker 默认开满 OpenCV 线程池
（= 核数），N workers × 核数 = 超订。实测 16 workers：默认线程 302 img/s → 置 1 后 403。

batch 与 nbs：`accum = round(nbs/batch)`，官方对 weight decay 做 `wd × batch × accum / nbs`
缩放（本仓库未实现）——只有 batch 整除 nbs 时因子才 = 1.0（nbs=64 → 16/32/64），
其余值会静默偏离官方语义。脚本会对非整除值告警。

用法:
    python scripts/bench_io.py                                   # 默认（约 5-8 分钟）
    python scripts/bench_io.py --loader-only                     # 只测数据管线（秒级）
    python scripts/bench_io.py --batch 16 32 64 --workers 8 12 16 20 --threads 0 1 2 --top 3
    python scripts/bench_io.py --data /path/to/coco --limit 20000
"""

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # 仓库根目录入 sys.path

import cv2
import torch
from torch.utils.data import DataLoader

from config.datasets import load_dataset
from config.train_config import TRAIN_CONFIG_PATH, load_train_config
from data.build import build_train_dataset
from data.loader import collate_fn, worker_init_fn
from model.yolo26 import CONFIG_PATH, YOLO26
from train.ema import ModelEMA
from train.loss import ComputeLoss
from train.optimizer import MuSGD, build_param_groups
from utils.logger import bold, get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)


def cpu_budget():
    """真实可用 CPU 数：cgroup v2 配额优先（affinity mask 常被容器夸大，本机 208 vs 实际 25）"""
    try:
        quota, period = open("/sys/fs/cgroup/cpu.max").read().split()
        if quota != "max":
            return max(1, int(int(quota) / int(period)))
    except Exception:  # noqa: BLE001 —— 非 cgroup v2 环境退回 affinity
        pass
    return len(os.sched_getaffinity(0))


def default_workers(step=8):
    """workers 候选：8/16/24…（步长 8，不超过 CPU 预算）"""
    return list(range(step, cpu_budget() + 1, step))


# cv2.getNumberOfCPUs() 而不是 cv2.getNumThreads()：data/dataset.py 导入时会把 cv2 线程数压到 1，
# 用 getNumThreads() 会读到 1，threads=0（还原库默认）就退化成 threads=1
_CV_DEFAULT, _TORCH_DEFAULT = cv2.getNumberOfCPUs(), torch.get_num_threads()  # 库默认（= 核数）


def make_worker_init(threads):
    """worker 初始化：RNG 播种 + 设置 OpenCV/torch 线程数

    threads=0 = 还原**库默认**（= 核数）。注意 worker_init_fn 内部会置 1，所以必须显式
    还原，否则 threads=0 与 threads=1 会退化成同一个配置（实测踩过）。

    ⚠ 子进程里调 cv2.setNumThreads 只在"父进程碰 cv2 之前 fork"时安全（OpenCV pthreads 池
    非 fork 安全，见 data/dataset.py 顶部注释）：本脚本的父进程只做 collate/搬运，不跑 cv2
    运算，因此这里的调用成立；若将来在父进程里加了 cv2 代码（可视化/存图），改为在父进程设定。
    """

    def _init(worker_id):
        worker_init_fn(worker_id)
        cv2.setNumThreads(threads or _CV_DEFAULT)
        torch.set_num_threads(threads or _TORCH_DEFAULT)

    return _init


def build_loader(ds, batch, workers, threads, seed=0):
    gen = torch.Generator().manual_seed(seed)  # 各配置同一抽样顺序，比较公平
    return DataLoader(ds, batch_size=batch, shuffle=True, num_workers=workers,
                      collate_fn=collate_fn, worker_init_fn=make_worker_init(threads),
                      generator=gen, pin_memory=True)


def bench_loader(ds, batch, workers, threads, warm, timed):
    """纯数据管线吞吐（不做任何 GPU 工作）"""
    dl = build_loader(ds, batch, workers, threads)
    for i, _ in enumerate(dl):
        if i + 1 >= warm:
            break
    n = 0
    t0 = time.monotonic()
    for i, (imgs, _t) in enumerate(dl):
        n += imgs.shape[0]
        if i + 1 >= timed:
            break
    el = time.monotonic() - t0
    del dl
    return n / el


def _step(model, loss_fn, opt, ema, imgs, targets, batch, cfg, device):
    """单个训练步（与 Trainer 主循环同口径，仅跳过 AMP/日志）"""
    imgs = imgs.to(device, non_blocking=True,
                   memory_format=torch.channels_last if cfg.channels_last else torch.preserve_format)
    targets = targets.to(device)
    preds = model(imgs)
    loss, _ = loss_fn(preds, targets, batch, cfg.imgsz)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
    opt.step()
    opt.zero_grad(set_to_none=True)
    ema.update(model)


def bench_e2e(ds, model, loss_fn, opt, ema, batch, workers, threads, warm, timed, cfg, device):
    """完整训练步吞吐（lr=0：权重不更新，只计时；避免发散切换代码路径）"""
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    dl = build_loader(ds, batch, workers, threads)
    for i, (imgs, targets) in enumerate(dl):
        if i + 1 >= warm:
            break
        _step(model, loss_fn, opt, ema, imgs, targets, batch, cfg, device)
    t0 = time.monotonic()
    nb = 0
    for imgs, targets in dl:
        _step(model, loss_fn, opt, ema, imgs, targets, batch, cfg, device)
        nb += 1
        if nb >= timed:
            break
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    el = time.monotonic() - t0
    del dl
    mem = torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0.0
    return nb / el, nb * batch / el, 1000 * el / nb, mem


def main():
    parser = argparse.ArgumentParser(description="training I/O throughput benchmark")
    parser.add_argument("--data", required=True,
                        help="dataset descriptor: a name in config/datasets/ (local/ wins) or a .yaml path")
    parser.add_argument("--batch", type=int, nargs="+", default=[8, 16, 32, 64],
                        help="batch sizes (end-to-end stage; 阶梯候选，超显存会自动 OOM 跳过)")
    parser.add_argument("--workers", type=int, nargs="+", default=None,
                        help="worker counts (default: 8/16/24... up to the CPU budget)")
    parser.add_argument("--threads", type=int, nargs="+", default=[0, 1, 2],
                        help="cv2/torch threads per worker (0 = library default)")
    parser.add_argument("--top", type=int, default=3, help="loader configs carried into the end-to-end stage")
    parser.add_argument("--batches", type=int, default=50, help="timed batches per end-to-end config")
    parser.add_argument("--warm", type=int, default=10, help="warmup batches per config")
    parser.add_argument("--limit", type=int, default=20000,
                        help="bench on the first N train images (0 = full; keeps page cache stable)")
    parser.add_argument("--loader-only", action="store_true", help="skip the end-to-end stage")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cfg = load_train_config(TRAIN_CONFIG_PATH)
    cfg.data = args.data
    try:
        spec = load_dataset(cfg.data)
    except (ValueError, FileNotFoundError) as e:
        parser.error(str(e))
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    if not args.workers:
        args.workers = default_workers()
    cores = cpu_budget()
    gpu = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    vram = torch.cuda.get_device_properties(device).total_memory // 2**20 if device.type == "cuda" else 0
    logger.info(bold(f"bench: {gpu} ({vram}MiB) · {cores} cores · imgsz {cfg.imgsz} · nbs {cfg.nbs} · "
                     f"limit {args.limit or 'full'})"))

    bad = [b for b in sorted(set(args.batch)) if cfg.nbs % b]
    if bad:
        logger.warning(f"batch {bad} 不整除 nbs={cfg.nbs} → accum 取整后 wd 因子 ≠ 1.0，与官方不等价"
                       f"（本仓库未实现 wd 缩放）；建议 {[b for b in (16, 32, 64, 128) if b <= cfg.nbs]}")

    ds = build_train_dataset(cfg, spec, "train", augment=True, limit=args.limit, progress=False)

    # ---- ① loader-only 全网格 ----
    logger.info("")
    logger.info(bold(f"[1/2] loader-only 网格（workers × threads，batch {max(args.batch)}）"))
    rows = []
    for workers in sorted(set(args.workers)):
        for threads in sorted(set(args.threads)):
            ips = bench_loader(ds, max(args.batch), workers, threads, args.warm, args.batches)
            rows.append((ips, workers, threads))
            logger.info(f"  workers {workers:>2} · threads {threads or 'default':>7}: {ips:>7.1f} img/s")
    rows.sort(reverse=True)
    top = rows[: max(args.top, 1)]
    logger.info(bold("  loader-only 前三: " + " · ".join(
        f"w{w}/t{t or 'def'} {ips:.0f} img/s" for ips, w, t in top)))
    if args.loader_only:
        return

    # ---- ② end-to-end ----
    logger.info("")
    logger.info(bold(f"[2/2] end-to-end（batch × loader 前三，各 {args.batches} batch）"))
    model = YOLO26(CONFIG_PATH, cfg.scale, cfg.imgsz, len(spec.names)).to(device).train()
    if cfg.channels_last:
        model.to(memory_format=torch.channels_last)
    head = model.model[-1]
    loss_fn = ComputeLoss(cfg, head, device)
    loss_fn.set_alpha(0, cfg.epochs)
    opt = MuSGD(build_param_groups(model, cfg, head), lr=0.0,
                momentum=cfg.momentum, muon_w=cfg.muon_w, sgd_w=cfg.sgd_w, ns_iters=cfg.ns_iters)
    ema = ModelEMA(model, decay=cfg.ema_decay, tau=cfg.ema_tau)

    e2e = []
    for batch in sorted(set(args.batch)):
        for _ips, workers, threads in top:
            try:
                its, ips, ms, mem = bench_e2e(ds, model, loss_fn, opt, ema, batch, workers, threads,
                                              args.warm, args.batches, cfg, device)
                e2e.append((batch, workers, threads, its, ips, ms, mem))
                logger.info(f"  batch {batch:>3} · workers {workers:>2} · threads {threads or 'default':>7}: "
                            f"{its:>5.2f} it/s · {ips:>6.1f} img/s · {ms:>4.0f} ms/step · peak {mem:>5.1f}G")
            except torch.cuda.OutOfMemoryError:
                logger.warning(f"  batch {batch:>3} · workers {workers:>2}: OOM (skip)")
                torch.cuda.empty_cache()

    if not e2e:
        return
    e2e.sort(key=lambda r: -r[4])
    logger.info("")
    logger.info(bold("end-to-end 排名（img/s）:"))
    for i, (batch, workers, threads, its, ips, ms, mem) in enumerate(e2e):
        logger.info(f"  {i + 1}. batch {batch:>3} · workers {workers:>2} · threads {threads or 'default':>7} · "
                    f"{ips:>6.1f} img/s · {its:>5.2f} it/s · peak {mem:>5.1f}G"
                    + ("  ← best" if i == 0 else ""))
    b, w, t, its, ips, _ms, _mem = e2e[0]
    logger.info("")
    logger.info(bold(f"loader-only 天花板 {top[0][0]:.0f} img/s  vs  端到端最佳 {ips:.0f} img/s"
                     + ("（差距大 = 瓶颈在主进程侧，加 worker 无用）"
                        if top[0][0] > 1.5 * ips else "")))
    logger.info(bold(f"推荐 train.yaml:  batch: {b}   workers: {w}"
                     + (f"   # 另需 worker 内置 cv2/torch 单线程（threads={t}）" if t else "")))
    logger.info(bold(f"              → 约 {ips:.1f} img/s · {its:.2f} it/s · "
                     f"{118287 / ips / 3600:.1f} h / 100 epochs"))


if __name__ == "__main__":
    main()
