"""训练中 pose 快速指标（OKS AP）：合成数据上的已知答案"""

import numpy as np

from train.metrics import FastPoseMetrics, _oks_matrix

K = 17


def _kpts(x, y, w, h, vis=2.0):
    out = []
    for i in range(K):
        out.append([x + (i % 5 + 0.5) / 5 * w, y + (i // 5 + 0.5) / 4 * h, vis])
    return np.array(out, np.float32)


def test_oks_matrix_perfect_and_offset():
    gt = _kpts(10, 10, 50, 50)
    box = np.array([[10, 10, 60, 60]], np.float32)
    sig = np.ones(K, np.float32) / K
    assert np.isclose(_oks_matrix(gt[None], gt[None], box, sig)[0, 0], 1.0, atol=1e-6)
    shifted = gt.copy()
    shifted[:, :2] += 20
    assert _oks_matrix(shifted[None], gt[None], box, sig)[0, 0] < 0.9


def test_fast_pose_metrics_perfect_match():
    gt = _kpts(10, 10, 50, 50)
    pred = gt.copy()
    pred[:, 2] = 1.0
    m = FastPoseMetrics(nc=1, kpt_shape=(K, 3))
    m.update(pred[None], np.array([0.9], np.float32), np.array([0], np.int64),
             gt[None], np.array([[10, 10, 60, 60]], np.float32), np.array([0], np.int64))
    out = m.compute()
    for k in ("mAP@50(P)", "mAP@[.5:.95](P)", "AR@20(P)"):
        assert out[k] == 1.0, (k, out[k])
    assert out["images"] == 1 and out["instances"] == 1


def test_fast_pose_metrics_ignores_zero_keypoint_gt():
    """num_keypoints==0 的标注按 pycocotools 口径忽略，不压低 AP"""
    gt = _kpts(10, 10, 50, 50)
    pred = gt.copy()
    pred[:, 2] = 1.0
    zero = np.zeros((K, 3), np.float32)
    m = FastPoseMetrics(nc=1, kpt_shape=(K, 3))
    m.update(pred[None], np.array([0.9], np.float32), np.array([0], np.int64),
             np.stack([gt, zero]), np.array([[10, 10, 60, 60], [0, 0, 1, 1]], np.float32),
             np.array([0, 0], np.int64))
    out = m.compute()
    assert out["mAP@50(P)"] == 1.0 and out["mAP@[.5:.95](P)"] == 1.0
    assert out["instances"] == 1


def test_oks_matrix_uses_annotation_area():
    """给定 COCO area 时覆盖框面积；更小的面积 -> 更低的 OKS"""
    gt = _kpts(10, 10, 50, 50)
    box = np.array([[10, 10, 60, 60]], np.float32)
    shifted = gt.copy()
    shifted[:, :2] += 20
    sig = np.ones(K, np.float32) / K
    oks_box = _oks_matrix(shifted[None], gt[None], box, sig)[0, 0]
    oks_area = _oks_matrix(shifted[None], gt[None], box, sig, gt_areas=np.array([500], np.float32))[0, 0]
    assert 0.0 < oks_area < oks_box < 1.0


def test_fast_pose_metrics_no_prediction():
    gt = _kpts(10, 10, 50, 50)
    m = FastPoseMetrics(nc=1, kpt_shape=(K, 3))
    m.update(np.zeros((0, K, 3), np.float32), np.zeros(0, np.float32), np.zeros(0, np.int64),
             gt[None], np.array([[10, 10, 60, 60]], np.float32), np.array([0], np.int64))
    out = m.compute()
    assert out["mAP@50(P)"] == 0.0 and out["mAP@[.5:.95](P)"] == 0.0
