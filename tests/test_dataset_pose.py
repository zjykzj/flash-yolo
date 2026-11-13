"""COCO-pose 数据/增强/批次契约：关键点与框逐行对齐、letterbox/flip、collate"""

import json
from pathlib import Path

import cv2
import numpy as np
import torch

from config.train_config import TrainConfig
from data.augment import flip_lr, letterbox_train
from data.coco import CocoDataset, CocoTrainDataset
from data.loader import collate_pose_fn

K = 17
FLIP_IDX = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]


def _kpts(x, y, w, h):
    out = []
    for i in range(K):
        out += [x + (i % 5 + 0.5) / 5 * w, y + (i // 5 + 0.5) / 4 * h, 2]
    return out


def _make_coco(tmp_path):
    root = Path(tmp_path)
    (root / "images").mkdir()
    img = np.zeros((80, 120, 3), np.uint8)
    cv2.imwrite(str(root / "images" / "a.jpg"), img)
    cv2.imwrite(str(root / "images" / "b.jpg"), img)
    ann = {
        "images": [{"id": 1, "file_name": "a.jpg"}, {"id": 2, "file_name": "b.jpg"},
                   {"id": 3, "file_name": "b.jpg"}],
        "categories": [{"id": 1, "name": "person", "keypoints": [f"k{i}" for i in range(K)],
                        "skeleton": []}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 50, 50], "area": 2500,
             "iscrowd": 0, "keypoints": _kpts(10, 10, 50, 50), "num_keypoints": K},
            # image 2 has a person box but no valid keypoints -> filtered out of pose training
            {"id": 2, "image_id": 2, "category_id": 1, "bbox": [5, 5, 30, 30], "area": 900,
             "iscrowd": 0, "keypoints": [0] * (3 * K), "num_keypoints": 0},
            {"id": 3, "image_id": 3, "category_id": 1, "bbox": [20, 20, 60, 40], "area": 2400,
             "iscrowd": 0, "keypoints": _kpts(20, 20, 60, 40), "num_keypoints": K},
        ],
    }
    path = root / "ann.json"
    path.write_text(json.dumps(ann))
    return path, root / "images"


def test_pose_dataset_alignment_and_filter(tmp_path):
    ann, imgs = _make_coco(tmp_path)
    cfg = TrainConfig(imgsz=64, mosaic=0.0, fliplr=0.0, mixup=0.0, cutmix=0.0)
    ds = CocoTrainDataset(cfg, ann, imgs, pose=True, kpt_shape=(K, 3), flip_idx=FLIP_IDX,
                          require_keypoints=True, augment=False)
    assert len(ds) == 2  # image 2 (num_keypoints=0) is outside the pose protocol
    assert [im["id"] for im in ds.images] == [1, 3]
    img, labels, kpts = ds[0]
    assert img.shape == (3, 64, 64)
    assert labels.shape == (1, 5) and kpts.shape == (1, K, 3)
    # letterbox 后框与关键点用同一 ratio/pad
    from data.preprocess import letterbox
    _, ratio, (pt, pl) = letterbox(np.zeros((80, 120, 3), np.uint8), new_shape=64)
    assert np.allclose(labels[0, 1:], [10 * ratio + pl, 10 * ratio + pt,
                                       60 * ratio + pl, 60 * ratio + pt], atol=1e-3)
    assert np.allclose(kpts[0, 0, :2], [15 * ratio + pl, 16.25 * ratio + pt], atol=1e-3)
    assert np.all(kpts[..., 0] >= labels[:, None, 1] - 1e-3)
    assert np.all(kpts[..., 0] <= labels[:, None, 3] + 1e-3)
    eval_ds = CocoDataset(ann, imgs, require_keypoints=True, kpt_shape=(K, 3))
    assert len(eval_ds) == 2 and eval_ds.keypoints(0).shape == (1, K, 3)
    assert len(eval_ds.targets(0, include_crowd=False)) == 1
    assert np.allclose(eval_ds.areas(0, include_crowd=False), [2500.0])


def test_pose_flip_lr_uses_flip_idx():
    kps = np.zeros((1, K, 3), np.float32)
    kps[0, :, 0] = np.arange(K) + 1
    kps[0, :, 1] = 5
    kps[0, :, 2] = 2
    out = flip_lr(np.zeros((10, 20, 3), np.uint8), np.zeros((0, 5), np.float32),
                  keypoints=kps, flip_idx=FLIP_IDX)[2]
    assert out.shape == kps.shape
    # 先镜像 x（W=20），再按 flip_idx 重排
    expected_x = (20 - (np.arange(K) + 1))[FLIP_IDX]
    assert np.allclose(out[0, :, 0], expected_x)
    assert np.allclose(out[0, :, 1], 5) and np.allclose(out[0, :, 2], 2)


def test_pose_augment_mosaic_and_flip(tmp_path):
    """mosaic + fliplr 全路径：关键点随框同步变换，行数一致、坐标有限"""
    ann, imgs = _make_coco(tmp_path)
    cfg = TrainConfig(imgsz=64, mosaic=1.0, fliplr=1.0, mixup=0.0, cutmix=0.0)
    ds = CocoTrainDataset(cfg, ann, imgs, pose=True, kpt_shape=(K, 3), flip_idx=FLIP_IDX,
                          require_keypoints=True, augment=True)
    for _ in range(3):
        img, labels, kpts = ds[0]
        assert img.shape == (3, 64, 64)
        assert len(labels) == len(kpts)
        assert np.isfinite(kpts).all()
        # 官方 apply_keypoints 口径：越界点坐标不 clip，但可见性置 0
        oob = ((kpts[..., 0] < 0) | (kpts[..., 0] > 64)
               | (kpts[..., 1] < 0) | (kpts[..., 1] > 64))
        assert np.all(kpts[oob][:, 2] == 0)


def test_pose_collate_alignment():
    batch = []
    for n in (1, 2):
        labels = np.array([[0, 10, 10, 20, 20]] * n, np.float32)
        kps = np.zeros((n, K, 3), np.float32)
        kps[..., 2] = 1
        batch.append((torch.zeros(3, 64, 64), labels, kps))
    imgs, targets, keypoints = collate_pose_fn(batch, (K, 3))
    assert imgs.shape == (2, 3, 64, 64)
    assert targets.shape == (3, 6) and keypoints.shape == (3, K, 3)
    assert np.allclose(targets[:, 0].numpy(), [0, 1, 1])
