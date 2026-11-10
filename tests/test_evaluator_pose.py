"""CocoEvaluator 的 keypoints（pose）口径：合成数据上的已知答案"""

import numpy as np

from eval.coco_evaluator import CocoEvaluator
from utils.engines import Detections

K = 17


def _gt_dict(gt_kpts):
    return {
        "images": [{"id": 1, "file_name": "a.jpg", "width": 100, "height": 100}],
        "categories": [{"id": 1, "name": "person",
                        "keypoints": [f"k{i}" for i in range(K)], "skeleton": []}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1,
                         "bbox": [10, 10, 60, 60], "area": 3600, "iscrowd": 0,
                         "keypoints": gt_kpts, "num_keypoints": K}],
    }


def _keypoints(offset=0.0, vis=2):
    """17 个落在 GT 框内的关键点：x,y,v 交错"""
    out = []
    for i in range(K):
        out += [10 + (i % 5) * 10 + offset, 10 + (i // 5) * 15 + offset, vis]
    return out


def _perfect_dets(vis=1.0):
    kpts = np.array(_keypoints(), np.float32).reshape(1, K, 3)
    kpts[..., 2] = vis
    return Detections(boxes=np.array([[10, 10, 70, 70]], np.float32),
                      scores=np.array([0.9], np.float32),
                      class_ids=np.array([0], np.int64), keypoints=kpts)


def test_perfect_match():
    """预测与 GT 完全重合 -> AP/AP50/AP75/AR 全 1.0"""
    ev = CocoEvaluator(gt=_gt_dict(_keypoints()), nc=1, iou_type="keypoints")
    ev.update(1, _perfect_dets())
    m = ev.compute()
    for k in ("AP", "AP50", "AP75", "AR", "AR50", "AR75"):
        assert m[k] == 1.0, f"{k} 应为 1.0，实际 {m[k]}"
    assert m["per_class"][0][0] == "person"
    assert m["per_class"][0][1:3] == (1, 1)


def test_no_prediction():
    """有 GT 无预测 -> AP = 0（空结果短路，不触发 pycocotools 崩溃）"""
    ev = CocoEvaluator(gt=_gt_dict(_keypoints()), nc=1, iou_type="keypoints")
    ev.update(1, Detections(boxes=np.zeros((0, 4), np.float32),
                            scores=np.zeros((0,), np.float32),
                            class_ids=np.zeros((0,), np.int64),
                            keypoints=np.zeros((0, K, 3), np.float32)))
    m = ev.compute()
    assert m["AP"] == 0.0 and m["AR"] == 0.0


def test_missing_keypoints_errors():
    """keypoints 评估器收到没有 keypoints 的 Detections -> 明确报错"""
    ev = CocoEvaluator(gt=_gt_dict(_keypoints()), nc=1, iou_type="keypoints")
    try:
        ev.update(1, Detections(boxes=np.array([[10, 10, 70, 70]], np.float32),
                                scores=np.array([0.9], np.float32),
                                class_ids=np.array([0], np.int64)))
    except ValueError as e:
        assert "keypoints" in str(e)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for missing keypoints")


def test_keypoint_offset_lowers_ap():
    """关键点整体偏移 -> OKS 下降，AP < 1（证明评估器真的在用关键点，不是只看框）"""
    ev = CocoEvaluator(gt=_gt_dict(_keypoints()), nc=1, iou_type="keypoints")
    dets = _perfect_dets()
    dets.keypoints[..., :2] += 25.0  # 偏移一个框宽
    ev.update(1, dets)
    m = ev.compute()
    assert m["AP"] < 1.0
