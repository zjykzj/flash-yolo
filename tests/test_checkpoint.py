"""checkpoint 验收：resume 往返逐位一致 / strict 回载 / best-last 落盘"""

import copy

import torch
import torch.nn as nn

from model.weights import load_weights
from model.yolo26 import YOLO26
from train.checkpoint import load_resume, save_best_last, save_resume
from config.train import TrainConfig
from train.ema import ModelEMA
from train.optimizer import MuSGD


def _step(model, opt, x, y):
    opt.zero_grad(set_to_none=True)
    loss = ((model(x) - y) ** 2).mean()
    loss.backward()
    opt.step()


def test_resume_roundtrip_identical(tmp_path):
    """3 步保存 -> 全新模型恢复 -> 再 2 步 == 连续 5 步（逐位一致）"""
    torch.manual_seed(0)
    x = torch.randn(8, 4)
    y = torch.randn(8, 4)

    def make():
        m = nn.Linear(4, 4, bias=False)  # muon 组只含 ndim∈{2,4} 参数（与 build_param_groups 口径一致）
        opt = MuSGD([{"params": m.parameters(), "muon": True, "wd": 0.0, "lr_mult": 1.0}], lr=0.01)
        return m, opt

    # 连续 5 步基线
    torch.manual_seed(0)
    m_ref, opt_ref = make()
    for _ in range(5):
        _step(m_ref, opt_ref, x, y)

    # 3 步保存 -> 首轮继续走完剩余 2 步 -> 恢复后走 2 步（两侧各 5 步对齐比较）
    torch.manual_seed(0)
    m, opt = make()
    ema = ModelEMA(m)
    for _ in range(3):
        _step(m, opt, x, y)
        ema.update(m)
    path = tmp_path / "resume.pt"
    save_resume(path, m, ema, opt, None, 3, 0.0, TrainConfig(), tmp_path)
    for _ in range(2):
        _step(m, opt, x, y)
        ema.update(m)

    m2, opt2 = make()
    ema2 = ModelEMA(m2)
    epoch, best, cfg, run_dir = load_resume(path, m2, ema2, opt2)
    assert epoch == 3 and best == 0.0
    for _ in range(2):
        _step(m2, opt2, x, y)
        ema2.update(m2)

    for (n1, p1), (n2, p2) in zip(m_ref.state_dict().items(), m2.state_dict().items()):
        assert n1 == n2 and torch.equal(p1, p2), f"resume 后参数不一致: {n1}"
    for (n1, p1), (n2, p2) in zip(ema.ema.state_dict().items(), ema2.ema.state_dict().items()):
        assert torch.equal(p1, p2), f"EMA 不一致: {n1}"
    print("  resume 往返逐位一致（模型 + EMA + 优化器状态）")


def test_best_last_weights(tmp_path):
    """best 仅在 is_best 时落盘；safetensors 可 strict 回载到 YOLO26"""
    model = YOLO26(scale="n")
    save_best_last(tmp_path, model, is_best=False)
    assert (tmp_path / "weights" / "last.safetensors").exists()
    assert not (tmp_path / "weights" / "best.safetensors").exists(), "非新高不应写 best"
    save_best_last(tmp_path, model, is_best=True)
    assert (tmp_path / "weights" / "best.safetensors").exists()

    fresh = YOLO26(scale="n")
    missing, unexpected = load_weights(fresh, tmp_path / "weights" / "best.safetensors", strict=True)
    assert not missing and not unexpected
    print("  best/last 落盘与 strict 回载正确")
