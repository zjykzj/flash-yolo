"""损失口径 parity：与本机 ultralytics（官方实现）逐项数值对齐（dev 依赖，需装 requirements-dev.txt）

固定种子构建模型与合成数据——每图 3 个互不相交、边长 ≥96px 的大目标（STAL 不触发、
候选池不共享、框边不与 anchor 中心重合），保证两个实现的分配结果一致；随后对比
o2m（topk 10）与 o2o（topk 7 -> 1）两分支的 box / cls / l1 三项与 ProgLoss 加权总损失。
差异只允许 fp32 舍入级（本仓库 grid 单位分配 vs 官方 px 单位分配、求和顺序不同）。

背景：cls 项曾用「全元素均值」（分母 B·N·nc ≈ 4.3e7）代替官方 Σtarget_scores（百量级），
梯度被稀释 ~10^6 倍——分类头 12 个 epoch 纹丝不动、验证 mAP 恒 0。本测试是这类
归一化口径漂移的回归闸门；L1 项同理（官方按 stride/imgsz 归一化后取 4 边均值）。
"""

from types import SimpleNamespace

import pytest
import torch

from config.train_config import TrainConfig
from model.build import DetectionModel
from train.loss import ComputeLoss

v8DetectionLoss = pytest.importorskip("ultralytics.utils.loss").v8DetectionLoss

IMGSZ = 320
BATCH = 2
# 每图 3 个目标：边长 ~104px、互不相交（间隔 >27px）、边坐标避开各层 anchor 中心
BOXES = [
    [(45.3, 40.7, 149.3, 144.7), (180.1, 45.9, 285.1, 150.9), (48.7, 190.3, 152.7, 294.3)],
    [(40.9, 48.1, 144.9, 152.1), (176.3, 36.7, 281.3, 141.7), (52.1, 186.9, 156.1, 294.1)],
]


def _rel(a, b):
    return abs(a - b) / max(abs(b), 1e-12)


def _case():
    torch.manual_seed(0)
    model = DetectionModel(scale="n").train()
    with torch.no_grad():
        preds = model(torch.randn(BATCH, 3, IMGSZ, IMGSZ))
    targets = torch.tensor(
        [[bi, k, *b] for bi, boxes in enumerate(BOXES) for k, b in enumerate(boxes)], dtype=torch.float32
    )
    return model, preds, targets


def _ref_setup(model, targets, cfg):
    """官方 v8DetectionLoss 的输入：batch dict（归一化 xywh）+ 假 feats（只用 shape 推导 imgsz 与 anchor）"""
    model.args = SimpleNamespace(box=cfg.box_gain, cls=cfg.cls_gain, dfl=cfg.dfl_gain, epochs=cfg.epochs)
    feats = [torch.zeros(BATCH, 1, IMGSZ // s, IMGSZ // s) for s in (8, 16, 32)]
    xywh = torch.stack([
        (targets[:, 2] + targets[:, 4]) / 2, (targets[:, 3] + targets[:, 5]) / 2,
        targets[:, 4] - targets[:, 2], targets[:, 5] - targets[:, 3],
    ], 1) / IMGSZ
    batch = {"batch_idx": targets[:, 0], "cls": targets[:, 1:2], "bboxes": xywh}
    return feats, batch


def test_loss_parity_vs_ultralytics():
    """两分支三项损失 + 加权总损失与官方数值一致（rel < 1e-4）"""
    cfg = TrainConfig()
    model, preds, targets = _case()
    loss_fn = ComputeLoss(cfg, model.model[-1], torch.device("cpu"))
    loss_fn.set_alpha(0, cfg.epochs)  # o2m 0.8 / o2o 0.2，与官方 E2ELoss 初值一致
    feats, batch = _ref_setup(model, targets, cfg)

    ref_totals = {}
    for branch, kw, name in (("one2many", {"tal_topk": 10}, "o2m"),
                             ("one2one", {"tal_topk": 7, "tal_topk2": 1}, "o2o")):
        ref = v8DetectionLoss(model, **kw)
        ref_loss, ref_items = ref.loss(
            {"boxes": preds[branch]["boxes"], "scores": preds[branch]["scores"], "feats": feats}, batch
        )
        ref_totals[name] = ref_loss.sum()

        # 单分支运行本仓库损失：items 即该分支的 box/cls/dfl（pre-gain per-image 均值）
        _, it = loss_fn({branch: preds[branch]}, targets, BATCH, IMGSZ)
        assert _rel(it[name] * BATCH, ref_totals[name].item()) < 1e-4, (name, it[name])
        assert _rel(it["box"] * cfg.box_gain, ref_items["box_loss"].item()) < 1e-4
        assert _rel(it["cls"] * cfg.cls_gain, ref_items["cls_loss"].item()) < 1e-4
        assert _rel(it["l1"] * cfg.dfl_gain, ref_items["l1_loss"].item()) < 1e-4

    # ProgLoss 加权总损失（双分支一次运行）
    total, _ = loss_fn(preds, targets, BATCH, IMGSZ)
    ref_combined = ref_totals["o2m"] * 0.8 + ref_totals["o2o"] * 0.2
    assert _rel(total.item(), ref_combined.item()) < 1e-4, (total.item(), ref_combined.item())
    print(f"  parity OK: o2m={ref_totals['o2m']:.4f} o2o={ref_totals['o2o']:.4f} total={total.item():.4f}")
