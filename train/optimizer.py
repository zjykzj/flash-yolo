"""MuSGD 优化器（论文 Sec 3.3.1）+ 参数分组

每步两个可加更新（Muon 组）:
    Muon: m <- beta*m + (1-beta)*g; u = beta*m + (1-beta)*g (Nesterov)
          u 重塑 2D -> Frobenius 归一 -> NS 迭代正交化 -> sqrt(max(1, rows/cols)) 缩放
          p -= lr*muon_w*u_orth
    SGD:  b <- momentum*b + (g + wd*p)（独立动量缓冲）; p -= lr*sgd_w*(momentum*b + g)
仅 SGD 部分含 weight decay。非 Muon 组走纯 Nesterov SGD。
NS 全程 fp32（GradScaler 已在 step 前 unscale），绝不在 autocast 下计算。
"""

import math

import torch

__all__ = ["MuSGD", "build_param_groups"]

NS_COEFFS = (3.4445, -4.7750, 2.0315)  # Newton-Schulz 5 次迭代系数（fp32 专调）


def build_param_groups(model, cfg, head):
    """参数分组（官方 yolo26n 配方口径）

    - muon 组:    ndim in {2,4}（线性矩阵 / 4D 卷积核），带 weight decay
    - 无衰减组:   ndim==1（bias）或 *.bn.weight
    - 其余权重组: 带 decay 的非 Muon 参数（罕见 1D 权重）
    - Detect.cv3 / one2one_cv3 参数 lr_mult=3（分类头微调加速）
    """
    head_ids = set()
    for branch in ("cv3", "one2one_cv3"):
        m = getattr(head, branch, None)
        if m is not None:
            for p in m.parameters():
                head_ids.add(id(p))

    buckets = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        lr_mult = 3.0 if id(p) in head_ids else 1.0
        muon = p.ndim in (2, 4)
        wd = 0.0 if (p.ndim == 1 or name.endswith(".bn.weight")) else cfg.weight_decay
        key = (muon, wd, lr_mult)
        buckets.setdefault(key, []).append(p)

    groups = []
    for (muon, wd, lr_mult), params in sorted(buckets.items()):
        groups.append(
            {"params": params, "muon": muon, "wd": wd, "lr_mult": lr_mult}
        )
    return groups


def _ortho(u2d, iters=5):
    """Newton-Schulz 松弛正交化（Muon 口径）

    u2d: (rows, cols) 2D 张量；rows > cols 时对转置做迭代后转回。
    输出奇异值落在 ~[0.5, 1.5]（非严格 {0,1} 投影，经验性松弛收敛），
    随后按原始矩阵方向缩放 sqrt(max(1, rows/cols))。
    """
    rows, cols = u2d.shape
    scale = math.sqrt(max(1.0, rows / cols))
    transposed = False
    if rows > cols:
        u2d = u2d.T
        transposed = True
    u2d = u2d / (u2d.norm() + 1e-7)
    a, b, c = NS_COEFFS
    for _ in range(iters):
        A = u2d @ u2d.T
        u2d = a * u2d + (b * A + c * (A @ A)) @ u2d
    if transposed:
        u2d = u2d.T
    return u2d * scale


class MuSGD(torch.optim.Optimizer):
    """Muon + SGD 混合优化器

    Args:
        params: build_param_groups 产出的分组（每组带 muon/wd/lr_mult 键）
        lr: 基础学习率（各组实际 lr = lr * lr_mult）
        momentum: 两个动量缓冲共用
        muon_w / sgd_w: 两更新分支的混合权重（官方 yolo26n: 0.528 / 0.674）
        ns_iters: Newton-Schulz 迭代次数
    """

    def __init__(self, params, lr, momentum=0.947, muon_w=0.528, sgd_w=0.674, nesterov=True, ns_iters=5):
        defaults = dict(lr=lr, momentum=momentum, muon_w=muon_w, sgd_w=sgd_w, nesterov=nesterov, ns_iters=ns_iters)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """单步更新（调用方负责 zero_grad；本类不清理梯度）"""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            momentum = group["momentum"]
            lr = group["lr"]
            nesterov = group.get("nesterov", True)
            wd = group.get("wd", 0.0)
            for p in group["params"]:
                g = p.grad
                if g is None:
                    continue
                if g.is_sparse:
                    raise RuntimeError("MuSGD does not support sparse gradients")
                if not torch.isfinite(g).all():
                    raise ValueError(f"non-finite gradient detected (MuSGD) — 训练发散，见 assigner clamp 修复")
                state = self.state[p]
                if group.get("muon", False):
                    if "muon_m" not in state:
                        state["muon_m"] = torch.zeros_like(p)
                        state["sgd_b"] = torch.zeros_like(p)
                    # ---- Muon 更新 ----
                    m = state["muon_m"]
                    m.mul_(momentum).add_(g, alpha=1 - momentum)  # m <- beta*m + (1-beta)*g
                    u = m.mul(momentum).add(g, alpha=1 - momentum) if nesterov else m
                    u2d = u.reshape(u.shape[0], -1) if u.ndim > 1 else u.reshape(1, -1)  # 防御 1D（分组契约 ndim∈{2,4}）
                    u_orth = _ortho(u2d, group["ns_iters"])
                    p.add_(u_orth.reshape_as(p), alpha=-lr * group["muon_w"])
                    # ---- SGD 更新（独立缓冲，wd 只在此生效）----
                    b = state["sgd_b"]
                    d = g.add(p, alpha=wd)
                    b.mul_(momentum).add_(d)
                    p.add_(b.mul(momentum).add_(d) if nesterov else b, alpha=-lr * group["sgd_w"])
                else:
                    # ---- 纯 Nesterov SGD ----
                    if "sgd_b" not in state:
                        state["sgd_b"] = torch.zeros_like(p)
                    b = state["sgd_b"]
                    d = g.add(p, alpha=wd)
                    b.mul_(momentum).add_(d)
                    p.add_(b.mul(momentum).add_(d) if nesterov else b, alpha=-lr)
        return loss
