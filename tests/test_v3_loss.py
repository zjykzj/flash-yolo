"""YOLOv3-tiny 损失验收：分派 / 匹配（形状 IoU + 责任格 + 忽略区）/ 数值口径 / 空批 / 梯度 / IoU-aware objectness"""

from types import SimpleNamespace

import pytest
import torch

from config.train_config import TrainConfig
from train.loss import build_loss
from train.loss.v3_detect import ComputeLossV3

_ANCHORS = [[[23, 27], [37, 58], [81, 82]], [[81, 82], [135, 169], [344, 319]]]


def _fake_head(nc=2):
    return SimpleNamespace(nc=nc, no=nc + 5, na=3,
                           stride=torch.tensor([16.0, 32.0]),
                           anchors=torch.tensor(_ANCHORS, dtype=torch.float32))


def _loss_fn(nc=2):
    return ComputeLossV3(TrainConfig(), _fake_head(nc), torch.device("cpu"))


def _raw(b, nc=2, h=2, w=2, fill=0.0):
    return torch.full((b, 3 * (5 + nc), h, w), fill)


def test_build_loss_dispatch():
    """按架构分派：yolov3-tiny -> ComputeLossV3（无 set_alpha）；未知架构报错"""
    from train.loss import ComputeLoss

    fn = build_loss("yolov3-tiny", TrainConfig(), _fake_head(2), torch.device("cpu"))
    assert isinstance(fn, ComputeLossV3)
    fn_silu = build_loss("yolov3-tiny-silu", TrainConfig(), _fake_head(2), torch.device("cpu"))
    assert isinstance(fn_silu, ComputeLossV3)
    assert fn.item_keys == ("box", "obj", "cls")
    assert not hasattr(fn, "set_alpha")
    y26 = build_loss("yolo26", TrainConfig(), SimpleNamespace(nc=80, stride=[8, 16, 32]), torch.device("cpu"))
    assert isinstance(y26, ComputeLoss) and y26.item_keys == ("box", "cls", "l1", "o2m", "o2o")
    with pytest.raises(ValueError, match="unknown arch"):
        build_loss("yolo5", TrainConfig(), _fake_head(2), torch.device("cpu"))
    print("  分派正确：yolov3-tiny -> ComputeLossV3 / yolo26 -> ComputeLoss")


def test_level_matching():
    """形状 IoU 选槽 + 责任 cell + 忽略区（1×1 网格手算）

    level0 锚 (23,27)/(37,58)/(81,82)、stride 16；GT 23×27 中心 (8,8) → cell(0,0)、槽位 0 应选中。
    预测框做成 GT 与三个锚的等价框：其余槽位 IoU 0.29/0.09 < 0.7 不进忽略区。
    """
    fn = _loss_fn()
    gt = torch.zeros(1, 1, 5)
    gt[0, 0] = torch.tensor([0.0, 8 - 11.5, 8 - 13.5, 8 + 11.5, 8 + 13.5])
    gt_mask = torch.ones(1, 1, dtype=torch.bool)
    pb = torch.tensor([[[8 - 11.5, 8 - 13.5, 8 + 11.5, 8 + 13.5],
                        [8 - 18.5, 8 - 29.0, 8 + 18.5, 8 + 29.0],
                        [8 - 40.5, 8 - 41.0, 8 + 40.5, 8 + 41.0]]])

    pos, ignore, t_box, t_cls = fn._level_targets(0, pb, 1, 1, gt, gt_mask)
    assert pos.shape == (1, 3)
    assert pos[0].tolist() == [True, False, False], pos[0]
    assert not ignore.any(), ignore
    assert torch.allclose(t_box[0, 0], gt[0, 0, 1:5])
    assert t_cls[0, 0, 0] == 1.0 and t_cls[0].sum() == 1.0

    # 忽略区：槽位 1 的预测框改成与 GT 几乎重合（IoU ≈ 0.92 ≥ 0.7）→ 置忽略（但不成为正样本）
    pb2 = pb.clone()
    pb2[0, 1] = torch.tensor([8 - 12.0, 8 - 14.0, 8 + 12.0, 8 + 14.0])
    pos2, ignore2, _, _ = fn._level_targets(0, pb2, 1, 1, gt, gt_mask)
    assert pos2[0].tolist() == [True, False, False]
    assert ignore2[0].tolist() == [False, True, False], ignore2[0]

    # padding GT（gt_mask=False）不产生正样本/忽略
    pos3, ignore3, _, _ = fn._level_targets(0, pb, 1, 1, gt, torch.zeros(1, 1, dtype=torch.bool))
    assert not pos3.any() and not ignore3.any()
    print("  匹配正确：形状 IoU 选槽 / 责任格 / 忽略区 / padding 屏蔽")


def test_loss_basic_and_empty():
    """基础数值：正样本存在时 box/cls > 0；空批时 box/cls 恰为 0、obj 为全负 BCE（有限非负）"""
    fn = _loss_fn()
    preds = [_raw(1, h=2, w=2), _raw(1, h=1, w=1)]
    targets = torch.tensor([[0, 0, 10, 10, 33, 37]], dtype=torch.float32)  # 23×27，命中 level0 槽位 0
    total, items = fn(preds, targets, 1, imgsz=32)
    assert torch.isfinite(total) and total > 0
    assert set(items) == {"box", "obj", "cls"}
    assert items["box"] > 0 and items["cls"] > 0 and items["obj"] > 0

    t_empty, it_empty = fn(preds, torch.zeros(0, 6), 1, imgsz=32)
    assert it_empty["box"] == 0.0 and it_empty["cls"] == 0.0
    assert torch.isfinite(t_empty) and t_empty > 0  # 全负样本的 obj BCE
    print(f"  基础数值正确: total={total.item():.3f}, items={items}")


def test_total_scales_with_batch():
    """total = Σ_图：同内容 batch=2 的 total == 两个 batch=1 之和（逐图归一化口径）"""
    fn = _loss_fn()
    t1 = torch.tensor([[0, 0, 10, 10, 33, 37]], dtype=torch.float32)
    t2 = torch.tensor([[0, 1, 40, 40, 90, 90]], dtype=torch.float32)
    total_a, _ = fn([_raw(1, h=2, w=2), _raw(1, h=1, w=1)], t1, 1, imgsz=32)
    total_b, _ = fn([_raw(1, h=2, w=2), _raw(1, h=1, w=1)], t2, 1, imgsz=32)

    t2b = torch.cat([t1, t2])
    t2b[1, 0] = 1.0
    total_2, _ = fn([_raw(2, h=2, w=2), _raw(2, h=1, w=1)], t2b, 2, imgsz=32)
    assert abs((total_2 - (total_a + total_b)).item()) < 1e-5, (total_2.item(), total_a.item(), total_b.item())
    print(f"  Σ_图 口径正确: batch2 {total_2.item():.4f} == {total_a.item():.4f} + {total_b.item():.4f}")


def test_gradient_reaches_backbone():
    """真实模型（nc=2，320 输入）：损失反向到达 backbone 与头卷积"""
    from model.build import build_model

    model = build_model("yolov3-tiny", imgsz=320, nc=2).train()
    head = model.model[-1]
    fn = ComputeLossV3(TrainConfig(), head, torch.device("cpu"))
    preds = model(torch.randn(1, 3, 320, 320))  # train 口径：逐级 raw 列表
    assert isinstance(preds, list) and preds[0].shape[1] == 3 * (5 + 2)
    targets = torch.tensor([[0, 0, 40, 40, 120, 120], [0, 1, 160, 160, 280, 280]], dtype=torch.float32)
    total, items = fn(preds, targets, 1, imgsz=320)
    total.backward()
    assert torch.isfinite(total)
    assert model.model[0].conv.weight.grad is not None, "backbone 应有梯度"
    assert head.m[0].weight.grad is not None, "头卷积应有梯度"
    assert all(torch.isfinite(p.grad).all() for p in head.parameters() if p.grad is not None)
    print(f"  梯度流正确: total={total.item():.3f}, items={items}")


def test_obj_iou_target_soft():
    """IoU-aware objectness（obj_iou_target）：正样本目标 = IoU(pred,gt) 且权重 ×5（闭式校验）

    raw 全 0 → obj logits = 0 → BCE(0, t) = ln2 对任意目标成立（与 IoU 具体值无关），
    不变量可精确手算：level0（2×2 网格、12 槽、1 正 11 负、无忽略）= (5+11)·ln2/12 = 4·ln2/3；
    level1（1×1、1 正 2 负）= (5+1+1)·ln2/3 = 7·ln2/3；
    items = 两级均值 = 11·ln2/6。box/cls 两模式逐位一致，total 差 = 5·ln2/3。
    """
    import math

    hard = _loss_fn()
    soft = ComputeLossV3(TrainConfig(obj_iou_target=True), _fake_head(2), torch.device("cpu"))
    preds = [_raw(1, h=2, w=2), _raw(1, h=1, w=1)]
    targets = torch.tensor([[0, 0, 10, 10, 33, 37]], dtype=torch.float32)  # 23×27，中心 (21.5, 23.5)
    t_h, it_h = hard(preds, targets, 1, imgsz=32)
    t_s, it_s = soft(preds, targets, 1, imgsz=32)
    ln2 = math.log(2.0)
    assert abs(it_h["obj"] - ln2) < 1e-6, it_h["obj"]
    assert abs(it_s["obj"] - 11 * ln2 / 6) < 1e-6, it_s["obj"]
    assert it_s["box"] == it_h["box"] and it_s["cls"] == it_h["cls"]
    assert abs((t_s - t_h).item() - 5 * ln2 / 3) < 1e-6, (t_s.item(), t_h.item())

    # 空 GT：无正样本 → 目标/权重两模式相同（退化一致）
    empty = torch.zeros(0, 6)
    t_eh, it_eh = hard(preds, empty, 1, imgsz=32)
    t_es, it_es = soft(preds, empty, 1, imgsz=32)
    assert it_es == it_eh and abs((t_es - t_eh).item()) < 1e-9
    print(f"  IoU-aware objectness 正确: obj {it_s['obj']:.6f}（硬 {it_h['obj']:.6f}，期望 11·ln2/6={11 * ln2 / 6:.6f}）")
