"""MuSGD 验收：NS 正交性 / 玩具收敛 / 参数分组 / AMP 兼容 / 梯度不清理"""

import torch
import torch.nn as nn

from model.build import DetectionModel
from config.train_config import TrainConfig
from train.lr import cosine_lr, linear_lr, set_epoch_lr, step_lr, warmup_lr
from train.optimizer import MuSGD, _ortho, build_param_groups


def test_ns_relaxed_orthogonality():
    """NS5 松弛正交化：奇异值落在经验收敛区间 [0.5, 1.5]（x 原始方向 scale），迭代稳定"""
    torch.manual_seed(0)
    # 宽矩阵（rows<=cols）：无转置，scale=1 -> 奇异值 [0.5, 1.5]
    b = torch.randn(8, 16)
    v = _ortho(b, iters=5)
    sv = torch.linalg.svdvals(v)
    assert sv.min() >= 0.5 - 1e-3 and sv.max() <= 1.5 + 1e-3, sv
    # 高矩阵（rows>cols）：转置迭代 + scale=sqrt(rows/cols) -> 奇异值 x scale
    a = torch.randn(16, 8)
    u = _ortho(a, iters=5)
    su = torch.linalg.svdvals(u)
    scale = (16 / 8) ** 0.5
    assert su.min() >= 0.5 * scale - 1e-3 and su.max() <= 1.5 * scale + 1e-3, su
    print("  NS5 松弛正交化正确（奇异值在 [0.5,1.5] 区间）")


def _toy_loss(model):
    x = torch.randn(16, 4)
    y = x @ torch.diag(torch.tensor([1.0, 2.0, 3.0, 4.0]))
    return ((model(x) - y) ** 2).mean()


def test_musgd_toy_convergence():
    """Muon 组与纯 SGD 组都在凸二次问题上显著下降"""
    for muon in (True, False):
        model = nn.Linear(4, 4, bias=False)
        init_loss = _toy_loss(model).item()
        opt = MuSGD([{"params": model.parameters(), "muon": muon, "wd": 0.0, "lr_mult": 1.0}], lr=0.02, momentum=0.9)
        for _ in range(500):
            opt.zero_grad(set_to_none=True)
            loss = _toy_loss(model)
            loss.backward()
            opt.step()
        final_loss = _toy_loss(model).item()
        ratio = final_loss / init_loss
        assert ratio < 0.5, f"muon={muon} 收敛不足: {init_loss:.2f} -> {final_loss:.2f} ({ratio:.2f})"
        print(f"  muon={muon}: {init_loss:.2f} -> {final_loss:.4f}")


def test_step_keeps_grads():
    """step 不清梯度（zero_grad 是训练器职责）"""
    model = nn.Linear(2, 2, bias=False)
    model.weight.data.fill_(1.0)
    loss = (model(torch.ones(1, 2)) ** 2).sum()
    loss.backward()
    opt = MuSGD([{"params": model.parameters(), "muon": True, "wd": 0.0, "lr_mult": 1.0}], lr=0.01)
    opt.step()
    assert model.weight.grad is not None and (model.weight.grad != 0).any()
    # Muon 组应有双缓冲
    st = opt.state[model.weight]
    assert "muon_m" in st and "sgd_b" in st
    print("  step 保留梯度，双缓冲就绪")


def test_build_param_groups():
    """YOLO26n 分组：muon=ndim{2,4} / bias与BN无衰减 / cv3·one2one_cv3 lr×3 / 全覆盖"""
    model = DetectionModel(scale="n")
    head = model.model[-1]
    groups = build_param_groups(model, TrainConfig(), head)
    n_grouped = sum(len(g["params"]) for g in groups)
    n_trainable = sum(1 for p in model.parameters() if p.requires_grad)
    assert n_grouped == n_trainable, f"{n_grouped} vs {n_trainable}"

    def owner_of(p):
        return next(g for g in groups if any(p is q for q in g["params"]))

    for g in groups:
        for p in g["params"]:
            if g["muon"]:
                assert p.ndim in (2, 4), f"muon 组混入 {p.ndim}D 参数"
    for name, p in model.named_parameters():
        if name.endswith(".bias") or name.endswith(".bn.weight"):
            assert owner_of(p)["wd"] == 0.0, f"{name} 不应带 decay"
    for branch in ("cv3", "one2one_cv3"):
        for p in getattr(head, branch).parameters():
            assert owner_of(p)["lr_mult"] == 3.0, f"{branch} 应 lr×3"
    n_muon = sum(len(g["params"]) for g in groups if g["muon"])
    print(f"  分组正确: {len(groups)} 组, muon 参数 {n_muon}, 可训参数 {n_trainable}")


def test_amp_compatibility():
    """autocast 下训练一步无 NaN"""
    model = nn.Linear(4, 4, bias=False)
    opt = MuSGD([{"params": model.parameters(), "muon": True, "wd": 0.0, "lr_mult": 1.0}], lr=0.01)
    x = torch.randn(8, 4)
    with torch.autocast("cpu"):
        loss = (model(x) ** 2).mean()
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    assert torch.isfinite(model.weight).all()
    print("  autocast 一步无 NaN")


def test_lr_schedules():
    assert warmup_lr(0.0, 3) == 0.0 and warmup_lr(3.0, 3) == 1.0
    assert abs(linear_lr(0, 245, 0.0054, 0.0495) - 0.0054) < 1e-12
    assert abs(linear_lr(245, 245, 0.0054, 0.0495) - 0.0054 * 0.0495) < 1e-12
    assert abs(cosine_lr(245, 245, 0.0054, 0.0495) - 0.0054 * 0.0495) < 1e-12
    # 台阶衰减（darknet steps 口径）：80%/90% 处各 ×0.1，无 lrf 终点因子
    assert abs(step_lr(0.0, 300, 0.001) - 0.001) < 1e-15
    assert abs(step_lr(0.79 * 300, 300, 0.001) - 0.001) < 1e-15
    assert abs(step_lr(0.80 * 300, 300, 0.001) - 0.0001) < 1e-15
    assert abs(step_lr(0.95 * 300, 300, 0.001) - 0.00001) < 1e-18
    m = nn.Linear(2, 2)
    opt = MuSGD([{"params": m.parameters(), "muon": False, "wd": 0.0, "lr_mult": 3.0}], lr=0.01)
    set_epoch_lr(opt, 0.001)
    assert abs(opt.param_groups[0]["lr"] - 0.003) < 1e-12, "lr_mult 应生效"
    print("  LR 调度正确")
