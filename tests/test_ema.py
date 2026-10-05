"""EMA 验收：滑动方向 / decay 爬升 / eval 副本"""

import torch
import torch.nn as nn

from train.ema import ModelEMA


def test_ema_tracks_model():
    """常数模型参数：EMA 单调逼近"""
    model = nn.Linear(2, 2)
    ema = ModelEMA(model, decay=0.9999, tau=2000)  # 先建副本（随机初始值），再把模型置常数
    model.weight.data.fill_(1.0)
    model.bias.data.fill_(0.5)

    prev = 0.0
    for _ in range(50):
        ema.update(model)
        v = ema.ema.weight.mean().item()
        assert 0.0 <= v <= 1.0, "EMA 值应在 [0,1] 内单调上升（fp32 下可舍入到 1.0）"
        assert v >= prev - 1e-12
        prev = v
    assert prev > 0.999, f"EMA 应逼近模型值: {prev:.4f}"
    print(f"  EMA 单调逼近: 50 步后 {prev:.6f}")


def test_ema_eval_model():
    model = nn.Linear(2, 2).train()
    ema = ModelEMA(model, decay=0.9999)
    out = ema.eval_model()
    assert not out.training, "EMA 副本应为 eval 模式"
    assert all(not p.requires_grad for p in out.parameters())
    print("  EMA 副本 eval + 冻结梯度")
