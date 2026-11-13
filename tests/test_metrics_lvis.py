"""LVIS 训练中快速指标：federated valid-class、frequency 分组、maxDets 语义"""

import json
from pathlib import Path

import cv2
import numpy as np

from data.coco import CocoDataset
from train.metrics import FastLvisMetrics
from train.validator import format_val_row


def _box(x, y, w=10, h=10):
    return [x, y, x + w, y + h]


def test_fast_lvis_perfect_detection_and_frequency_groups():
    m = FastLvisMetrics(nc=3, task="detect", frequencies={0: "r", 1: "c", 2: "f"})
    boxes = np.array([_box(0, 0), _box(20, 0), _box(40, 0)], np.float32)
    m.update(boxes, np.array([0.9, 0.8, 0.7], np.float32), np.array([0, 1, 2], np.int64),
             boxes.copy(), np.array([0, 1, 2], np.int64), np.ones(3, bool))
    out = m.compute()
    assert out["mAP@50"] == 1.0 and out["mAP@[.5:.95]"] == 1.0
    assert out["APr"] == 1.0 and out["APc"] == 1.0 and out["APf"] == 1.0
    assert out["AR@100"] == 1.0 and out["instances"] == 3


def test_fast_lvis_ignores_detections_in_invalid_classes():
    """federated 未标注类上的检测必须忽略，不能记成 FP"""
    m = FastLvisMetrics(nc=1, task="detect", frequencies={0: "c"})
    box = np.array([_box(10, 10)], np.float32)
    # 图 1：有效 GT + 正确检测
    m.update(box, np.array([0.9], np.float32), np.array([0], np.int64),
             box.copy(), np.array([0], np.int64), np.ones(1, bool))
    # 图 2：该类未标注（valid=False），高置信检测应被忽略
    m.update(box, np.array([0.99], np.float32), np.array([0], np.int64),
             np.zeros((0, 4), np.float32), np.zeros(0, np.int64), np.zeros(1, bool))
    out = m.compute()
    assert out["mAP@[.5:.95]"] == 1.0
    assert out["images"] == 2 and out["instances"] == 1


def test_fast_lvis_mask_metric():
    m = FastLvisMetrics(nc=1, task="segment", frequencies={0: "c"})
    box = np.array([_box(10, 10)], np.float32)
    mask = np.ones((1, 16, 16), np.uint8)
    m.update(box, np.array([0.9], np.float32), np.array([0], np.int64),
             box.copy(), np.array([0], np.int64), np.ones(1, bool))
    m.update_mask(mask, np.array([0.9], np.float32), np.array([0], np.int64),
                  mask.copy(), np.array([0], np.int64), np.ones(1, bool))
    out = m.compute()
    assert out["mAP@[.5:.95](M)"] == 1.0
    assert out["APc(M)"] == 1.0


def test_format_val_row_lvis_includes_frequency_columns():
    metrics = {"images": 1, "instances": 2, "P": 0.1, "R": 0.2, "mAP@50": 0.3,
               "mAP@[.5:.95]": 0.4, "APr": 0.05, "APc": 0.06, "APf": 0.07, "AR@100": 0.5}
    row = format_val_row(metrics, lvis=True)
    assert "0.0500" in row and "0.0600" in row and "0.0700" in row


def test_coco_dataset_lvis_valid_classes_and_frequencies(tmp_path):
    root = Path(tmp_path)
    imgs = root / "images"
    imgs.mkdir()
    cv2.imwrite(str(imgs / "a.jpg"), np.zeros((80, 120, 3), np.uint8))
    ann = {
        "images": [{"id": 1, "file_name": "a.jpg", "neg_category_ids": [2],
                    "not_exhaustive_category_ids": [3]}],
        "categories": [{"id": 1, "name": "a", "frequency": "r"},
                       {"id": 2, "name": "b", "frequency": "c"},
                       {"id": 3, "name": "c", "frequency": "f"}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [1, 1, 10, 10],
                         "area": 100, "iscrowd": 0}],
    }
    path = root / "ann.json"
    path.write_text(json.dumps(ann))
    ds = CocoDataset(path, imgs)
    assert ds.valid_classes(0).tolist() == [True, True, False]
    assert ds.frequencies() == {0: "r", 1: "c", 2: "f"}
