"""学习率调度：warmup + 线性/余弦/台阶衰减（t 为 epoch 浮点进度，每 batch 粒度）"""

import math

__all__ = ["warmup_lr", "warmup_momentum", "linear_lr", "cosine_lr", "step_lr", "set_epoch_lr"]


def warmup_lr(t, warmup_epochs):
    """t < warmup_epochs 时线性爬升 0 -> 1，之后恒 1"""
    return min(t / max(warmup_epochs, 1e-9), 1.0)


def warmup_momentum(t, warmup_epochs, momentum, warmup_momentum=0.8):
    """warmup 期间动量线性爬升（官方训练循环 np.interp(ni, [0,nw], [warmup_momentum, momentum])）"""
    return warmup_momentum + (momentum - warmup_momentum) * warmup_lr(t, warmup_epochs)


def linear_lr(t, epochs, lr0, lrf):
    """线性衰减：lr0 -> lr0*lrf"""
    return lr0 * (1 - t / epochs * (1 - lrf))


def cosine_lr(t, epochs, lr0, lrf):
    """余弦衰减：lr0 -> lr0*lrf"""
    return lr0 * lrf + 0.5 * (lr0 - lr0 * lrf) * (1 + math.cos(math.pi * t / epochs))


def step_lr(t, epochs, lr0, step_fracs=(0.8, 0.9), gamma=0.1):
    """台阶衰减（darknet `policy=steps` 口径）：进度每越过一个 frac 就 ×gamma（默认 80% / 90% 处各降 10×）

    无 lrf 终点因子（darknet 无此概念；终值 = lr0 × gamma^len(step_fracs)）。
    """
    drops = sum(1 for f in step_fracs if t >= f * epochs)
    return lr0 * gamma ** drops


def set_epoch_lr(optimizer, base_lr, momentum=None):
    """按组 lr_mult 写入各组 lr（每 batch 调用）；momentum 非 None 时同时写入动量（warmup 用）"""
    for g in optimizer.param_groups:
        g["lr"] = base_lr * g.get("lr_mult", 1.0)
        if momentum is not None:
            g["momentum"] = momentum
