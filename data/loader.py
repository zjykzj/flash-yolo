"""训练数据加载的共享件：cv2 线程契约 / 跨 fork 计数 / worker 播种 / collate / 训练集 mixin

COCO 与 YOLO 两种训练集（data/coco.py、data/yolo.py）共用本模块——
增强入口、corrupt 计数与 worker 线程约定只允许有一份实现（两处必然漂移）。
"""

import multiprocessing

import cv2
import numpy as np
import torch

from data.augment import augment, letterbox_train

__all__ = ["TrainMixin", "SharedCounter", "collate_fn", "worker_init_fn"]

_worker_rng = None  # worker 进程内全局 RNG（worker_init_fn 播种，多进程下各 worker 独立）

# OpenCV 线程数必须在**导入时**（= 任何 cv2 并行运算与第一次 fork 之前）由父进程设定，子进程只继承：
# OpenCV 的 pthreads 线程池非 fork 安全 —— 父进程只要跑过一次 cv2 运算（本地 = 验证集 imread +
# 增强抽样网格 imwrite），线程池就按默认核数（25）初始化；此后再 fork 的 worker 若调用
# cv2.setNumThreads()，会去 join 一批子进程里并不存在的线程，永久阻塞在 futex（实测 16/16 worker
# 死锁、训练卡在 epoch 2 第一个 batch 前；epoch 1 的 fork 早于父进程碰 cv2，因此毫发无损 —— 时序竞态）。
# 父进程侧设 1 的代价为零：imread+letterbox 407(1 线程) vs 396(25 线程) img/s，JPEG 解码本就单线程。
cv2.setNumThreads(1)


class SharedCounter:
    """跨 fork 共享的计数器（DataLoader worker 里 +1，父进程可读）

    只在失败路径写、每 epoch 读一次，因此 lock=False（省锁；并发 +1 丢一两次对诊断无影响）。
    走 fork 继承（Linux 默认），与仓库其它 fork 约定一致（见文件顶部 cv2 线程契约）。
    """

    def __init__(self):
        self._v = multiprocessing.Value("i", 0, lock=False)

    def bump(self):
        self._v.value += 1

    @property
    def value(self):
        return self._v.value


class TrainMixin:
    """训练数据集共用行为：增强入口 / close_mosaic / 惰性 corrupt 计数

    子类需提供：cfg（duck-typing）、use_augment、load_image(idx) -> (BGR, labels)、_corrupt(SharedCounter)
    """

    def __getitem__(self, idx):
        img, labels = self.load_image(idx)
        if self.use_augment:
            return augment(img, labels, self.load_image, self.cfg, _rng(), idx, len(self))
        return letterbox_train(img, labels, self.cfg.imgsz)

    def close_mosaic(self):
        """最后 N epoch：清零 mosaic/mixup/copy_paste 概率（HSV/flip/perspective 保留）"""
        self.cfg.mosaic = self.cfg.mixup = self.cfg.copy_paste = 0.0

    @property
    def n_corrupt(self):
        return self._corrupt.value


def worker_init_fn(worker_id):
    """每 worker 独立播种 RNG（抽样流由 DataLoader 的 seed+epoch generator 保证可复现）

    并收紧 worker 内线程数：DataLoader 是多进程，若每个 worker 都用 OpenCV/torch 的默认
    线程池（= 核数），N 个 worker 会开出 N×核数 条线程互相抢占。实测（25 核，batch 64）：
    16 workers × 默认 25 线程 = 302 img/s；置 1 后 403 img/s（+34%）；20 workers × 1 = 496 img/s
    （+64%）。加载器是训练吞吐的瓶颈（GPU 利用率仅 ~14%），这条直接换来 epoch 提速。

    注意 cv2 的线程数**不在这里设**：它是 fork 不安全的（见文件顶部注释），改由父进程在
    导入时设定、子进程继承；若要核对，用 cv2.getNumThreads() 而不是加一行 setNumThreads。
    """
    global _worker_rng
    seed = int(torch.initial_seed() % (2**32))
    _worker_rng = np.random.default_rng(seed + worker_id)
    torch.set_num_threads(1)


def _rng():
    """主进程用默认 RNG；worker 进程用 worker_init_fn 播种的 RNG"""
    global _worker_rng
    return _worker_rng if _worker_rng is not None else np.random.default_rng()


def collate_fn(batch):
    """-> (imgs (B,3,H,W) float32, targets (N,6) [batch_idx, cls, x1, y1, x2, y2])"""
    imgs, labels = zip(*batch)
    targets = []
    for bi, lbs in enumerate(labels):
        if len(lbs):
            targets.append(np.concatenate([np.full((len(lbs), 1), bi, np.float32), lbs], axis=1))
    tgts = torch.from_numpy(np.concatenate(targets)) if targets else torch.zeros((0, 6))
    return torch.stack(imgs), tgts
