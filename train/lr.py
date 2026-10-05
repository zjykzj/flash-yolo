"""学习率调度：warmup + 线性/余弦衰减（t 为 epoch 浮点进度，每 batch 粒度）"""

import math

__all__ = ["warmup_lr", "linear_lr", "cosine_lr", "set_epoch_lr"]


def warmup_lr(t, warmup_epochs):
    """t < warmup_epochs 时线性爬升 0 -> 1，之后恒 1"""
    return min(t / max(warmup_epochs, 1e-9), 1.0)


def linear_lr(t, epochs, lr0, lrf):
    """线性衰减：lr0 -> lr0*lrf"""
    return lr0 * (1 - t / epochs * (1 - lrf))


def cosine_lr(t, epochs, lr0, lrf):
    """余弦衰减：lr0 -> lr0*lrf"""
    return lr0 * lrf + 0.5 * (lr0 - lr0 * lrf) * (1 + math.cos(math.pi * t / epochs))


def set_epoch_lr(optimizer, base_lr):
    """按组 lr_mult 写入各组 lr（每 batch 调用）"""
    for g in optimizer.param_groups:
        g["lr"] = base_lr * g.get("lr_mult", 1.0)
