"""分割损失 parity：与官方 v8SegmentationLoss 逐项数值对齐（dev 依赖 ultralytics）

固定种子构建 yolo26-seg（train 模式）与合成数据；GT 掩码由同一批框在 proto 网格上填充矩形。
对比 overlap=True/False 两种掩码契约下 o2m / o2o 两分支的 box/cls/l1/seg 四项与分支总损失。
官方 semseg 辅助项在本仓库未实现（Proto26 前向不产出 semantic），parity 只比较 seg 主项。
"""

from types import SimpleNamespace

import pytest
import torch

from config.train_config import TrainConfig
from model.build import build_model
from train.loss import ComputeLossSeg

v8SegmentationLoss = pytest.importorskip("ultralytics.utils.loss").v8SegmentationLoss

IMGSZ = 320
BATCH = 2
MASK_RATIO = 4
BOXES = [
    [(45.3, 40.7, 149.3, 144.7), (180.1, 45.9, 285.1, 150.9), (48.7, 190.3, 152.7, 294.3)],
    [(40.9, 48.1, 144.9, 152.1), (176.3, 36.7, 281.3, 141.7), (52.1, 186.9, 156.1, 294.1)],
]


def _rel(a, b):
    return abs(a - b) / max(abs(b), 1e-12)


def _case(grad=False):
    torch.manual_seed(0)
    model = build_model("yolo26-seg", "n", IMGSZ, nc=80).train()
    x = torch.randn(BATCH, 3, IMGSZ, IMGSZ)
    preds = model(x) if grad else None
    if preds is None:
        with torch.no_grad():
            preds = model(torch.randn(BATCH, 3, IMGSZ, IMGSZ))
    targets = torch.tensor(
        [[bi, k, *b] for bi, boxes in enumerate(BOXES) for k, b in enumerate(boxes)], dtype=torch.float32
    )
    return model, preds, targets


def _rect_masks(targets, overlap):
    """GT 框 -> proto 网格矩形掩码；overlap=True: (B,h,w) 实例索引；False: (N,h,w)"""
    mh = mw = IMGSZ // MASK_RATIO

    def box(row):
        x1, y1, x2, y2 = (row[2:] * (mw / IMGSZ)).tolist()
        return (max(int(round(x1)), 0), max(int(round(y1)), 0),
                min(int(round(x2)), mw), min(int(round(y2)), mh))

    rows = []
    for row in targets:
        m = torch.zeros(mh, mw, dtype=torch.int32)
        x1, y1, x2, y2 = box(row)
        m[y1:y2, x1:x2] = 1
        rows.append(m)
    if overlap:
        out = []
        for bi in range(BATCH):
            m = torch.zeros(mh, mw, dtype=torch.int32)
            for k, row in enumerate(targets[targets[:, 0] == bi]):
                x1, y1, x2, y2 = box(row)
                m[y1:y2, x1:x2] = k + 1
            out.append(m)
        return torch.stack(out)
    return torch.stack(rows)


def _ref_batch(targets, masks):
    xywh = torch.stack([
        (targets[:, 2] + targets[:, 4]) / 2, (targets[:, 3] + targets[:, 5]) / 2,
        targets[:, 4] - targets[:, 2], targets[:, 5] - targets[:, 3],
    ], 1) / IMGSZ
    return {"batch_idx": targets[:, 0], "cls": targets[:, 1:2], "bboxes": xywh, "masks": masks.float()}


@pytest.mark.parametrize("overlap", [True, False])
def test_seg_loss_parity_vs_ultralytics(overlap):
    cfg = TrainConfig(overlap_mask=overlap, mask_ratio=MASK_RATIO)
    model, preds, targets = _case()
    masks = _rect_masks(targets, overlap)
    batch = _ref_batch(targets, masks)
    loss_fn = ComputeLossSeg(cfg, model.model[-1], torch.device("cpu"))
    loss_fn.set_alpha(0, cfg.epochs)

    feats = [torch.zeros(BATCH, 1, IMGSZ // s, IMGSZ // s) for s in (8, 16, 32)]
    model.args = SimpleNamespace(box=cfg.box_gain, cls=cfg.cls_gain, dfl=cfg.dfl_gain,
                                 epochs=cfg.epochs, overlap_mask=overlap)

    for branch, kw, name in (("one2many", {"tal_topk": 10}, "o2m"),
                             ("one2one", {"tal_topk": 7, "tal_topk2": 1}, "o2o")):
        ref_loss_fn = v8SegmentationLoss(model, **kw)
        ref_loss, ref_items = ref_loss_fn.loss(
            {"boxes": preds[branch]["boxes"], "scores": preds[branch]["scores"],
             "mask_coefficient": preds[branch]["mask_coefficient"], "proto": preds[branch]["proto"],
             "feats": feats}, batch)
        _, it = loss_fn({branch: preds[branch]}, targets, BATCH, IMGSZ, masks=masks)
        assert _rel(it[name] * BATCH, ref_loss.sum().item()) < 1e-4, (name, it[name], ref_loss.sum().item())
        assert _rel(it["box"] * cfg.box_gain, ref_items["box_loss"].item()) < 1e-4
        assert _rel(it["cls"] * cfg.cls_gain, ref_items["cls_loss"].item()) < 1e-4
        assert _rel(it["l1"] * cfg.dfl_gain, ref_items["l1_loss"].item()) < 1e-4
        assert _rel(it["seg"] * cfg.box_gain, ref_items["seg_loss"].item()) < 1e-4, \
            (name, it["seg"] * cfg.box_gain, ref_items["seg_loss"].item())
        print(f"  {name} overlap={overlap}: seg={it['seg']:.6f} total={it[name]:.6f}")


def test_seg_loss_gradients_reach_both_mask_heads():
    """o2m/o2o 的 cv4 与 Proto26 都从掩码项拿到梯度（proto 只经 o2m 回流）"""
    cfg = TrainConfig(overlap_mask=True)
    model, preds, targets = _case(grad=True)
    masks = _rect_masks(targets, True)
    loss_fn = ComputeLossSeg(cfg, model.model[-1], torch.device("cpu"))
    model.zero_grad(set_to_none=True)
    total, items = loss_fn(preds, targets, BATCH, IMGSZ, masks=masks)
    total.backward()
    head = model.model[-1]
    assert items["seg"] > 0
    assert head.cv4[0][0].conv.weight.grad is not None
    assert head.one2one_cv4[0][0].conv.weight.grad is not None
    assert head.proto.cv1.conv.weight.grad is not None


def test_seg_loss_empty_foreground_is_finite():
    cfg = TrainConfig(overlap_mask=True)
    model, preds, _ = _case()
    loss_fn = ComputeLossSeg(cfg, model.model[-1], torch.device("cpu"))
    empty = torch.zeros((0, 6), dtype=torch.float32)
    masks = torch.zeros(BATCH, IMGSZ // MASK_RATIO, IMGSZ // MASK_RATIO, dtype=torch.int32)
    total, items = loss_fn(preds, empty, BATCH, IMGSZ, masks=masks)
    assert torch.isfinite(total)
    assert items["seg"] == 0.0
