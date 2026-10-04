"""验收3：CocoEvaluator 在合成数据上指标正确（已知答案）"""

import json

import numpy as np

from eval.coco_evaluator import CocoEvaluator
from utils.engine import Detections


def _make_ann_file(tmp_path, images, annotations):
    """构造临时 COCO 标注 json"""
    ann = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": i, "name": f"c{i}"} for i in range(80)],
    }
    path = tmp_path / "instances_test.json"
    path.write_text(json.dumps(ann))
    return str(path)


def test_perfect_match(tmp_path):
    """预测与 GT 完全重合 -> 全指标 = 1.0（含 small/large 面积分组）"""
    ann_file = _make_ann_file(
        tmp_path,
        images=[{"id": 1, "file_name": "a.jpg", "width": 640, "height": 640}],
        annotations=[
            {"id": 1, "image_id": 1, "category_id": 0, "bbox": [10, 10, 10, 10], "area": 100, "iscrowd": 0},  # small
            {"id": 2, "image_id": 1, "category_id": 1, "bbox": [50, 50, 200, 300], "area": 60000, "iscrowd": 0},  # large
        ],
    )
    ev = CocoEvaluator(ann_file)
    ev.update(
        1,
        Detections(
            boxes=np.array([[10, 10, 20, 20], [50, 50, 250, 350]], dtype=np.float32),
            scores=np.array([0.99, 0.98], dtype=np.float32),
            class_ids=np.array([0, 1], dtype=np.int64),
        ),
    )
    m = ev.compute()
    for k in ("mAP@[.5:.95]", "mAP@50", "mAP@75"):
        assert m[k] == 1.0, f"{k} 应为 1.0，实际 {m[k]}"
    assert m["mAP_small"] == 1.0
    assert m["mAP_large"] == 1.0


def test_no_prediction(tmp_path):
    """有 GT 无预测 -> mAP = 0.0"""
    ann_file = _make_ann_file(
        tmp_path,
        images=[{"id": 1, "file_name": "a.jpg", "width": 640, "height": 640}],
        annotations=[{"id": 1, "image_id": 1, "category_id": 0, "bbox": [10, 10, 90, 90], "area": 8100, "iscrowd": 0}],
    )
    ev = CocoEvaluator(ann_file)
    ev.update(
        1,
        Detections(
            boxes=np.zeros((0, 4), dtype=np.float32),
            scores=np.zeros((0,), dtype=np.float32),
            class_ids=np.zeros((0,), dtype=np.int64),
        ),
    )
    m = ev.compute()
    assert m["mAP@[.5:.95]"] == 0.0


def test_wrong_class(tmp_path):
    """类别错误 -> mAP = 0.0"""
    ann_file = _make_ann_file(
        tmp_path,
        images=[{"id": 1, "file_name": "a.jpg", "width": 640, "height": 640}],
        annotations=[{"id": 1, "image_id": 1, "category_id": 0, "bbox": [10, 10, 90, 90], "area": 8100, "iscrowd": 0}],
    )
    ev = CocoEvaluator(ann_file)
    ev.update(
        1,
        Detections(
            boxes=np.array([[10, 10, 100, 100]], dtype=np.float32),
            scores=np.array([0.99], dtype=np.float32),
            class_ids=np.array([1], dtype=np.int64),  # GT 是类别 0
        ),
    )
    m = ev.compute()
    assert m["mAP@[.5:.95]"] == 0.0
