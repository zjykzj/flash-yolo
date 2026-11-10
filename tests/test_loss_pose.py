"""YOLO26-pose 损失 parity：与本机 ultralytics `PoseLoss26` 逐项数值对齐（dev 依赖）

固定随机种子构建 Pose26 与合成关键点数据；对比 o2m/o2o 加权总损失与 o2o 分支的
pose/kobj/rle 三项（官方 items 只回 o2o；本仓库 items 是两分支 per-image 均值之和）。
"""

from types import SimpleNamespace

import pytest
import torch

from config.train_config import TrainConfig
from model.build import DetectionModel, YOLO26_POSE_CONFIG_PATH
from train.assigner import padded_gt
from train.loss import ComputeLossPose

PoseLoss26 = pytest.importorskip("ultralytics.utils.loss").PoseLoss26
E2ELoss = pytest.importorskip("ultralytics.utils.loss").E2ELoss

IMGSZ = 320
BATCH = 2
# 与 tests/test_loss_parity.py 同款：每图 3 个互不相交、边长 ~104px 的大目标，
# 边坐标避开 anchor 中心，保证两个实现的分配结果一致（否则 STAL/候选池差异会污染 parity）。
BOXES = [
    [(45.3, 40.7, 149.3, 144.7), (180.1, 45.9, 285.1, 150.9), (48.7, 190.3, 152.7, 294.3)],
    [(40.9, 48.1, 144.9, 152.1), (176.3, 36.7, 281.3, 141.7), (52.1, 186.9, 156.1, 294.1)],
]
K = 17


def _keypoints(box):
    x1, y1, x2, y2 = box
    out = []
    for i in range(K):
        out += [x1 + (i % 5 + 0.5) / 5 * (x2 - x1),
                y1 + (i // 5 + 0.5) / 4 * (y2 - y1), 2.0]
    return out


def _case():
    torch.manual_seed(0)
    model = DetectionModel(YOLO26_POSE_CONFIG_PATH, scale="n").train()
    boxes = [(bi, b) for bi, bs in enumerate(BOXES) for b in bs]
    targets = torch.tensor([[bi, 0, *b] for bi, b in boxes], dtype=torch.float32)
    keypoints = torch.tensor([_keypoints(b) for _, b in boxes], dtype=torch.float32).reshape(-1, K, 3)
    imgs = torch.randn(BATCH, 3, IMGSZ, IMGSZ)
    with torch.no_grad():
        preds = model(imgs)
        feats = model.forward_feats(imgs)
    return model, preds, feats, targets, keypoints


def _cfg():
    return TrainConfig(box_gain=7.5, cls_gain=0.5, dfl_gain=6.0, pose_gain=24.0,
                       kobj_gain=4.0, rle_gain=1.0, prog_alpha_init=0.8,
                       topk=10, topk_o2o=7, topk2=1)


def _official_batch(targets, keypoints):
    x1, y1, x2, y2 = targets[:, 2], targets[:, 3], targets[:, 4], targets[:, 5]
    xywh = torch.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], 1) / IMGSZ
    kp = keypoints.clone()
    kp[..., 0] /= IMGSZ
    kp[..., 1] /= IMGSZ
    return {"batch_idx": targets[:, 0].view(-1, 1), "cls": targets[:, 1].view(-1, 1),
            "bboxes": xywh, "keypoints": kp}


def _rel(a, b):
    return abs(a - b) / max(abs(b), 1e-12)


def test_pose_loss_parity_vs_ultralytics():
    cfg = _cfg()
    model, preds, feats, targets, keypoints = _case()
    loss_fn = ComputeLossPose(cfg, model.model[-1], torch.device("cpu"))
    loss, _ = loss_fn(preds, targets, BATCH, IMGSZ, keypoints=keypoints)

    model.args = SimpleNamespace(box=cfg.box_gain, cls=cfg.cls_gain, dfl=cfg.dfl_gain,
                                 epochs=cfg.epochs, pose=cfg.pose_gain, kobj=cfg.kobj_gain,
                                 rle=cfg.rle_gain)
    # 官方 v8PoseLoss 用 `kpt_shape == [17,3]` 判定 COCO OKS sigmas/RLE 权重；
    # 本仓库 head 存 tuple，临时改成 list 让官方实现走同一分支（仅测试适配）。
    model.model[-1].kpt_shape = [17, 3]
    for br in ("one2many", "one2one"):
        preds[br]["feats"] = feats
    e2e = E2ELoss(model, loss_fn=PoseLoss26)
    e2e.o2m, e2e.o2o = cfg.prog_alpha_init, 1.0 - cfg.prog_alpha_init
    with torch.no_grad():
        ref, ref_items = e2e(preds, _official_batch(targets, keypoints))
    assert _rel(loss.item(), ref.sum().item()) < 1e-4, (loss.item(), ref.sum().item())

    # o2o 分支的 pose/kobj/rle 三项（官方 items 带增益；本仓库 items 是 pre-gain）
    anchors, strides = loss_fn._anchors(IMGSZ)
    gt, gt_mask = padded_gt(targets, BATCH)
    _, _, _, a = loss_fn._branch_terms(
        preds["one2one"]["boxes"], preds["one2one"]["scores"], anchors, strides,
        gt, gt_mask, BATCH, IMGSZ, one2one=True)
    _, extra = loss_fn._extra_branch_terms(
        "one2one", preds["one2one"], a, None, targets, strides, BATCH, IMGSZ, keypoints=keypoints)
    assert _rel(float(extra["pose"] * cfg.pose_gain), ref_items["pose_loss"].item()) < 1e-4
    assert _rel(float(extra["kobj"] * cfg.kobj_gain), ref_items["kobj_loss"].item()) < 1e-4
    assert _rel(float(extra["rle"] * cfg.rle_gain), ref_items["rle_loss"].item()) < 1e-4
    print(f"  pose parity OK: total={loss.item():.4f} pose={ref_items['pose_loss']:.4f} "
          f"kobj={ref_items['kobj_loss']:.4f} rle={ref_items['rle_loss']:.4f}")


def test_pose_loss_backward():
    cfg = _cfg()
    model, preds, _, targets, keypoints = _case()
    loss_fn = ComputeLossPose(cfg, model.model[-1], torch.device("cpu"))
    loss, items = loss_fn(preds, targets, BATCH, IMGSZ, keypoints=keypoints)
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert all(torch.isfinite(torch.as_tensor(v)) for v in items.values())
