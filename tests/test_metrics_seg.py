"""训练中快速 Mask 指标：完美/错误/部分 IoU、空输入、box 指标不受影响"""

import numpy as np

from train.metrics import FastMetrics

H = W = 16


def _blank():
    return np.zeros((H, W), np.uint8)


def _rect(x1, y1, x2, y2):
    m = _blank()
    m[y1:y2, x1:x2] = 1
    return m


def _register(m, det_masks, scores, cls, gt_masks, gt_cls):
    """走 update（框占位，只为 n_images/instances）+ update_mask 的真实入口"""
    n = len(det_masks)
    m.update(np.zeros((n, 4), np.float32), np.asarray(scores, np.float32),
             np.asarray(cls, np.int64), np.zeros((len(gt_masks), 4), np.float32),
             np.asarray(gt_cls, np.int64))
    m.update_mask(np.stack(det_masks), np.asarray(scores, np.float32),
                  np.asarray(cls, np.int64), np.stack(gt_masks), np.asarray(gt_cls, np.int64))


def test_mask_perfect_matches():
    m = FastMetrics(nc=2, task="segment")
    gt = [_rect(1, 1, 6, 6), _rect(9, 9, 14, 14), _rect(1, 9, 5, 13)]
    _register(m, gt, [0.9, 0.8, 0.7], [0, 0, 1], gt, [0, 0, 1])
    out = m.compute()
    assert out["mAP@[.5:.95](M)"] == 1.0 and out["mAP@50(M)"] == 1.0
    assert out["P(M)"] == 1.0 and out["R(M)"] == 1.0
    # 框指标仍存在且独立（这里给的是占位框，指标为 0）
    assert out["mAP@[.5:.95]"] == 0.0


def test_mask_iou_partial():
    m = FastMetrics(nc=1, task="segment")
    gt = [_rect(0, 0, 10, 10)]
    det = [_rect(0, 0, 6, 10)]  # 面积 60/100 -> IoU 0.6
    _register(m, det, [0.9], [0], gt, [0])
    out = m.compute()
    assert out["mAP@50(M)"] == 1.0
    assert abs(out["mAP@[.5:.95](M)"] - 0.3) < 1e-9  # 仅 0.50/0.55/0.60 三个阈值命中


def test_mask_wrong_is_zero():
    m = FastMetrics(nc=1, task="segment")
    _register(m, [_rect(0, 0, 4, 4)], [0.9], [0], [_rect(10, 10, 14, 14)], [0])
    out = m.compute()
    assert out["mAP@[.5:.95](M)"] == 0.0 and out["mAP@50(M)"] == 0.0


def test_mask_empty_inputs():
    m = FastMetrics(nc=1, task="segment")
    m.update(np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, np.int64),
             np.zeros((0, 4), np.float32), np.zeros(0, np.int64))
    m.update_mask(np.zeros((0, H, W), np.uint8), np.zeros(0, np.float32), np.zeros(0, np.int64),
                  np.zeros((0, H, W), np.uint8), np.zeros(0, np.int64))
    out = m.compute()
    assert out["mAP@[.5:.95](M)"] == 0.0 and out["images"] == 1
    # 空 val（n_images=0）也返回完整键
    empty = FastMetrics(nc=1, task="segment").compute()
    assert "mAP@[.5:.95](M)" in empty and "P(M)" in empty
