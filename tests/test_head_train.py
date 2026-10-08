"""训练模式头部 + 权重保存验收

- 训练模式：model.train() 下 forward 返回双分支原始输出 dict
- o2o detach：o2o-only 损失反向时 backbone 无梯度、o2o 头有梯度（detach 特征输入而非输出）
- eval 模式：与改造前行为一致（推理/导出/评估路径不受影响）
- save_weights：safetensors 往返 strict 加载后参数一致
- bias_init 的 cls 先验随训练 imgsz 走
"""

import math

import torch
import pytest

from model.weights import load_weights, save_weights
from model.yolo26 import YOLO26


def _build(scale="n"):
    return YOLO26(scale=scale)


def test_train_mode_shapes():
    """训练模式下双分支原始输出形状"""
    model = _build().train()
    x = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        out = model(x)
    assert set(out) == {"one2many", "one2one"}, f"keys: {list(out)}"
    for key in ("one2many", "one2one"):
        assert out[key]["boxes"].shape == (1, 4, 8400), f"{key} boxes: {out[key]['boxes'].shape}"
        assert out[key]["scores"].shape == (1, 80, 8400), f"{key} scores: {out[key]['scores'].shape}"
    print("  train-mode 双分支形状正确 (1,4,8400)/(1,80,8400)")


def test_o2o_feature_detach():
    """o2o 分支梯度隔离：backbone 梯度只经 o2m 回流"""
    model = _build().train()
    head = model.model[-1]
    x = torch.randn(1, 3, 640, 640)

    # 只对 o2o 输出求损失：backbone 与 o2m 头无梯度，o2o 头有梯度
    out = model(x)
    out["one2one"]["boxes"].sum().backward()
    assert model.model[0].conv.weight.grad is None, "o2o-only 损失不应回传 backbone"
    assert head.cv2[0][0].conv.weight.grad is None, "o2o-only 损失不应回传 o2m 头"
    assert head.one2one_cv2[0][0].conv.weight.grad is not None, "o2o 头自身应有梯度"

    # 只对 o2m 输出求损失：backbone 有梯度，o2o 头无梯度（detach 阻断）
    model.zero_grad(set_to_none=True)
    out = model(x)
    out["one2many"]["boxes"].sum().backward()
    assert model.model[0].conv.weight.grad is not None, "o2m 损失应回传 backbone"
    assert head.cv2[0][0].conv.weight.grad is not None, "o2m 头应有梯度"
    assert head.one2one_cv2[0][0].conv.weight.grad is None, "detach 应阻断 o2m 损失进入 o2o 头"
    print("  o2o detach 梯度方向正确")


def test_eval_mode_unchanged():
    """eval 模式行为与改造前一致"""
    model = _build().eval()
    head = model.model[-1]
    x = torch.randn(1, 3, 640, 640)

    head.end2end = True
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 300, 6), f"E2E 输出形状: {out.shape}"

    head.end2end = False
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 84, 8400), f"o2m 原始输出形状: {out.shape}"
    print("  eval 模式两种路径形状不变")


def test_save_weights_roundtrip(tmp_path):
    """save_weights -> load_weights(strict) 参数逐位一致"""
    model = _build().eval()
    path = tmp_path / "roundtrip.safetensors"
    save_weights(model, path)

    fresh = _build()
    missing, unexpected = load_weights(fresh, path, strict=True)
    assert not missing and not unexpected
    for (n1, p1), (n2, p2) in zip(model.state_dict().items(), fresh.state_dict().items()):
        assert n1 == n2 and torch.equal(p1, p2), f"参数不一致: {n1}"
    print(f"  safetensors 往返一致（{len(model.state_dict())} 个张量）")


def test_bias_init_follows_imgsz():
    """cls 先验随训练 imgsz 缩放：bias = log(5 / nc / (imgsz/stride)²)，默认 640 = 官方口径"""
    head640 = _build().model[-1]  # 默认 imgsz=640
    head320 = YOLO26(scale="n", imgsz=320).model[-1]
    for i, s in enumerate((8, 16, 32)):  # 三档 stride 逐档核对
        want640 = math.log(5 / 80 / (640 / s) ** 2)
        want320 = math.log(5 / 80 / (320 / s) ** 2)
        assert abs(head640.cv3[i][2].bias[0].item() - want640) < 1e-6, f"stride {s}: 640 口径偏移"
        assert abs(head320.cv3[i][2].bias[0].item() - want320) < 1e-6, f"stride {s}: 320 口径偏移"
    # o2o 头同款初始化；box 头偏置恒 2.0（与 imgsz 无关）
    assert torch.equal(head320.cv3[0][2].bias, head320.one2one_cv3[0][2].bias)
    assert (head320.cv2[0][2].bias == 2.0).all()
    print(f"  bias_init 随 imgsz 缩放正确：stride8 cls bias {head320.cv3[0][2].bias[0].item():.4f}（320）"
          f" vs {head640.cv3[0][2].bias[0].item():.4f}（640）")
