"""损失验收：梯度流 / 空批有限 / ×batch 缩放 / ProgLoss 调度"""

import torch

from model.yolo26 import YOLO26
from config.train import TrainConfig
from train.loss import ComputeLoss


def _preds(model, batch=1):
    x = torch.randn(batch, 3, 640, 640)
    return x, model(x)


def test_loss_gradient_flow():
    """双分支损失反向：backbone 与两个头都有梯度（o2m 路径 + o2o 自身路径）"""
    model = YOLO26(scale="n").train()
    head = model.model[-1]
    cfg = TrainConfig()
    loss_fn = ComputeLoss(cfg, head, torch.device("cpu"))
    x, preds = _preds(model)
    targets = torch.tensor([[0, 5, 10, 10, 500, 500]], dtype=torch.float32)

    total, items = loss_fn(preds, targets, batch_size=1, imgsz=640)
    total.backward()
    assert torch.isfinite(total), f"loss NaN: {total}"
    assert model.model[0].conv.weight.grad is not None, "backbone 应有梯度（经 o2m）"
    assert head.cv2[0][0].conv.weight.grad is not None, "o2m 头应有梯度"
    assert head.one2one_cv2[0][0].conv.weight.grad is not None, "o2o 头应有梯度"
    assert all(v > 0 for v in (items["o2m"], items["o2o"]))
    print(f"  梯度流正确, total={total.item():.3f}, items={items}")


def test_loss_empty_batch():
    """无 GT 批：损失有限（cls 目标全零）"""
    model = YOLO26(scale="n").train()
    head = model.model[-1]
    loss_fn = ComputeLoss(TrainConfig(), head, torch.device("cpu"))
    with torch.no_grad():
        _, preds = _preds(model)
    total, items = loss_fn(preds, torch.zeros(0, 6), batch_size=1, imgsz=640)
    assert torch.isfinite(total) and total.item() >= 0
    assert items["box"] == 0.0 and items["dfl"] == 0.0
    print(f"  空批有限: total={total.item():.3f}")


def test_loss_batch_scaling():
    """同一图 batch=2 的损失 ≈ 2x batch=1（梯度口径线性）

    注意区间：官方三项损失共用分母 Σt 且 clamp(min=1)。init 时单 GT 的
    ciou_max ≈ 0.004 < 1（大 GT 覆盖小预测框），Σt 落进 clamp 区间则非线性
    （官方同款行为）；本测试用 8 个 ~50px GT 保证 Σt ≫ 1。
    """
    model = YOLO26(scale="n").train()
    head = model.model[-1]
    loss_fn = ComputeLoss(TrainConfig(), head, torch.device("cpu"))
    with torch.no_grad():
        x1, preds1 = _preds(model, batch=1)
        x2 = torch.cat([x1, x1])
        preds2 = model(x2)
    boxes = [(40, 40), (120, 40), (200, 40), (280, 40), (40, 120), (120, 120), (200, 120), (280, 120)]
    t1 = torch.tensor([[0, k, x, y, x + 50, y + 50] for k, (x, y) in enumerate(boxes)], dtype=torch.float32)
    t2b = t1.clone()
    t2b[:, 0] = 1.0
    t2 = torch.cat([t1, t2b])

    total1, _ = loss_fn(preds1, t1, batch_size=1, imgsz=640)
    total2, _ = loss_fn(preds2, t2, batch_size=2, imgsz=640)
    ratio = (total2 / total1).item()
    assert abs(ratio - 2.0) < 1e-4, f"batch 缩放比例 {ratio:.4f} != 2"
    print(f"  ×batch 缩放正确: {ratio:.4f}")


def test_degenerate_head_no_nan_grad():
    """回归：o2o 头退化场景（logits 极负 + 框退化）反向无 NaN 梯度

    实测根因：align = s^0.5 * ciou^6 在 s->0 且 ciou->0 时反向为 inf*0=NaN，
    经 Muon 动量缓冲毒化 o2o 权重（损失仍有限 -> 静默死亡）。修复后必须全有限。
    """
    from types import SimpleNamespace

    loss_fn = ComputeLoss(TrainConfig(), SimpleNamespace(nc=80, stride=[8, 16, 32]), torch.device("cpu"))
    for branch in ("one2many", "one2one"):
        boxes = torch.zeros(1, 4, 8400, requires_grad=True)  # 框退化为锚点
        scores = torch.full((1, 80, 8400), -100.0, requires_grad=True)  # sigmoid 下溢为精确 0
        preds = {"one2many": {"boxes": boxes, "scores": scores},
                 "one2one": {"boxes": boxes, "scores": scores}}
        targets = torch.tensor([[0, 5, 10, 10, 500, 500]], dtype=torch.float32)
        total, _ = loss_fn(preds, targets, batch_size=1, imgsz=640)
        total.backward()
        assert torch.isfinite(total), "损失应有限"
        # 修复前：s^(-0.5)=inf 与 ciou^6=0 相乘 -> scores.grad 出现 NaN
        if scores.grad is not None:
            assert torch.isfinite(scores.grad).all(), f"{branch} scores 梯度 NaN"
        if boxes.grad is not None:
            assert torch.isfinite(boxes.grad).all(), f"{branch} boxes 梯度 NaN"
        print(f"  {branch}: 退化场景反向全有限")


def test_progloss_schedule():
    """ProgLoss alpha 调度：0 -> 0.8, 中点 -> 0.45, 末 epoch -> 0.1"""
    from types import SimpleNamespace

    cfg = TrainConfig(prog_alpha_init=0.8, prog_alpha_final=0.1)
    head = SimpleNamespace(nc=80, stride=[8, 16, 32])  # 调度测试不接触前向
    loss_fn = ComputeLoss(cfg, head, torch.device("cpu"))
    loss_fn.set_alpha(0, 245)
    assert abs(loss_fn.alpha - 0.8) < 1e-9
    loss_fn.set_alpha(122, 245)
    assert abs(loss_fn.alpha - 0.45) < 1e-9, loss_fn.alpha
    loss_fn.set_alpha(244, 245)
    assert abs(loss_fn.alpha - 0.1) < 1e-9
    print("  ProgLoss 调度正确（0.8 -> 0.45 -> 0.1）")
