"""训练数据集验收：类别映射口径 / crowd 剔除 / limit / letterbox 标签映射 / collate"""

import json

import cv2
import numpy as np
import pytest
import torch

from data.coco import CocoDataset
from data.dataset import CocoTrainDataset, collate_fn
from config.train import TrainConfig


def _make_fixture(tmp_path):
    """3 图 2 类（非连续 category_id）+ 1 crowd 标注的迷你 COCO"""
    data_dir = tmp_path / "data"
    img_dir = data_dir / "train2017"
    ann_dir = data_dir / "annotations"
    img_dir.mkdir(parents=True)
    ann_dir.mkdir(parents=True)

    for name in ("a.jpg", "b.jpg", "c.jpg"):
        cv2.imwrite(str(img_dir / name), np.zeros((64, 64, 3), np.uint8))

    anns = {
        "images": [
            {"id": 1, "file_name": "a.jpg", "width": 64, "height": 64},
            {"id": 2, "file_name": "b.jpg", "width": 64, "height": 64},
            {"id": 3, "file_name": "c.jpg", "width": 64, "height": 64},
        ],
        "annotations": [
            {"id": 10, "image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20], "iscrowd": 0},
            {"id": 11, "image_id": 1, "category_id": 3, "bbox": [30, 30, 10, 10], "iscrowd": 1},
            {"id": 12, "image_id": 2, "category_id": 3, "bbox": [0, 0, 8, 8], "iscrowd": 0},
        ],
        "categories": [{"id": 1, "name": "cat"}, {"id": 3, "name": "dog"}],
    }
    with open(ann_dir / "instances_train2017.json", "w") as f:
        json.dump(anns, f)
    return data_dir


def test_category_mapping_consistent_with_coco(tmp_path):
    """与 CocoDataset 的类别映射口径逐项一致"""
    data_dir = _make_fixture(tmp_path)
    cfg = TrainConfig(data_dir=str(data_dir))
    ds = CocoTrainDataset(cfg, split="train2017")
    ref = CocoDataset(str(data_dir), split="train2017")
    assert ds.cat_id_to_idx == ref.cat_id_to_idx, f"{ds.cat_id_to_idx} vs {ref.cat_id_to_idx}"
    print("  类别映射与 CocoDataset 一致", ds.cat_id_to_idx)


def test_crowd_excluded_from_train(tmp_path):
    """iscrowd=1 训练侧剔除；CocoDataset（val 口径）保留"""
    data_dir = _make_fixture(tmp_path)
    ds = CocoTrainDataset(TrainConfig(data_dir=str(data_dir)), split="train2017")
    ref = CocoDataset(str(data_dir), split="train2017")
    assert len(ds.labels[0]) == 1, f"img1 训练标签应剔除 crowd: {ds.labels[0]}"
    assert len(ref.targets(0)) == 2, "CocoDataset 应保留 crowd（val 口径）"
    assert ds.labels[0][0][0] == 0 and list(ds.labels[0][0][1:]) == [10, 10, 30, 30]
    assert len(ds.labels[2]) == 0, "img3 无标签应为空数组"
    print("  crowd 剔除正确，训练/验证口径分离")


def test_limit_and_len(tmp_path):
    data_dir = _make_fixture(tmp_path)
    ds = CocoTrainDataset(TrainConfig(data_dir=str(data_dir)), split="train2017", limit=2)
    assert len(ds) == 2
    assert ds.images[0]["id"] == 1
    print("  limit 切片正确")


def test_letterbox_labels(tmp_path):
    """augment=False 路径：等比缩放 + 灰边，标签随 ratio/pad 变换"""
    data_dir = _make_fixture(tmp_path)
    ds = CocoTrainDataset(TrainConfig(data_dir=str(data_dir)), split="train2017", augment=False)
    tensor, labels = ds[0]
    assert tensor.shape == (3, 640, 640)
    # 64x64 -> 640x640: ratio=10, pad=0; [10,10,30,30] -> [100,100,300,300]
    assert len(labels) == 1
    np.testing.assert_allclose(labels[0], [0, 100, 100, 300, 300], atol=1e-3)
    print("  letterbox 标签映射正确:", labels[0])


def test_collate(tmp_path):
    data_dir = _make_fixture(tmp_path)
    ds = CocoTrainDataset(TrainConfig(data_dir=str(data_dir)), split="train2017", augment=False)
    imgs, targets = collate_fn([ds[0], ds[1], ds[2]])
    assert imgs.shape == (3, 3, 640, 640)
    # img0 1 框, img1 1 框, img2 0 框 -> (2, 6)
    assert targets.shape == (2, 6), f"targets: {targets.shape}"
    assert (targets[:, 0].numpy() == [0, 1]).all(), "batch_idx 错误"
    print("  collate 形状与 batch_idx 正确")
