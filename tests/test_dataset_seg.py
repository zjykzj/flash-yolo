"""分割数据管线：多边形解析/合并、掩码栅格化、样本与 collate 契约、增强几何、YOLO 拒绝"""

import json

import cv2
import numpy as np
import pytest
import torch

from config.train_config import TrainConfig
from data.augment import augment, flip_lr, letterbox_train, resize_long_side
from data.build import build_train_dataset
from data.coco import CocoDataset, CocoTrainDataset
from data.loader import collate_fn
from data.segments import annotation_polygon, merge_polygons, rasterize_segments
from config.datasets import load_dataset

SIZE = 64


def _rect(x1, y1, x2, y2):
    return [x1, y1, x2, y1, x2, y2, x1, y2]


def _make_dataset(tmp_path, annotations):
    root = tmp_path / "seg"
    (root / "images" / "train").mkdir(parents=True)
    (root / "annotations").mkdir(parents=True)
    img = np.full((SIZE, SIZE, 3), 200, np.uint8)
    cv2.rectangle(img, (8, 8), (40, 40), (50, 50, 200), -1)
    cv2.imwrite(str(root / "images" / "train" / "0001.jpg"), img)
    images = [{"id": 1, "file_name": "0001.jpg", "width": SIZE, "height": SIZE}]
    (root / "annotations" / "instances_train.json").write_text(json.dumps(
        {"images": images, "annotations": annotations,
         "categories": [{"id": 1, "name": "a"}, {"id": 2, "name": "b"}]}))
    desc = tmp_path / "seg.yaml"
    desc.write_text(f"format: coco\npath: {root}\nnames: [a, b]\n"
                    "train: {images: images/train, ann: annotations/instances_train.json}\n", encoding="utf-8")
    return desc, root


def _ann(aid, cls, seg, bbox=None):
    bbox = bbox or [8, 8, 32, 32]
    return {"id": aid, "image_id": 1, "category_id": cls,
            "bbox": bbox, "area": bbox[2] * bbox[3], "iscrowd": 0, "segmentation": seg}


def test_annotation_polygon_and_merge():
    poly, fallback = annotation_polygon(_ann(1, 1, [_rect(8, 8, 40, 40)]))
    assert not fallback and poly.shape == (4, 2)
    merged = merge_polygons([np.array([[0, 0], [10, 0], [10, 10]]),
                             np.array([[20, 20], [30, 20], [30, 30]])])
    assert merged.ndim == 2 and merged.shape[1] == 2 and merged.shape[0] >= 6
    # RLE dict -> bbox 矩形 fallback
    poly, fallback = annotation_polygon(_ann(2, 1, {"size": [SIZE, SIZE], "counts": "abc"}))
    assert fallback and poly.shape == (4, 2)
    # 无 bbox 的非法标注 -> None
    assert annotation_polygon({"segmentation": []}) == (None, False)


def test_scan_and_sample_contract(tmp_path):
    desc, _ = _make_dataset(tmp_path, [
        _ann(1, 1, [_rect(8, 8, 40, 40)]),
        _ann(2, 2, {"size": [SIZE, SIZE], "counts": "abc"}, bbox=[4, 4, 10, 10]),
    ])
    spec = load_dataset(desc)
    cfg = TrainConfig(imgsz=SIZE)
    ds = build_train_dataset(cfg, spec, "train", augment=False, task="segment")
    assert ds.n_seg_fallback == 1
    assert len(ds.segments) == 1 and len(ds.segments[0]) == 2
    img, labels, segments = ds.load_image(0)
    assert img.shape == (SIZE, SIZE, 3) and labels.shape == (2, 5) and len(segments) == 2
    # 框由多边形包围盒导出（第二个是 fallback 矩形）
    assert np.allclose(labels[0, 1:], [8, 8, 40, 40])
    assert np.allclose(labels[1, 1:], [4, 4, 14, 14])
    # 全 background 的未标注图不在这里构造；100% fallback 必须报错
    desc2, _ = _make_dataset(tmp_path / "allfb", [_ann(1, 1,
        {"size": [SIZE, SIZE], "counts": "abc"}, bbox=[4, 4, 10, 10])])
    with pytest.raises(ValueError, match="fell back to bbox rectangles"):
        build_train_dataset(cfg, load_dataset(desc2), "train", augment=False, task="segment")


def test_rasterize_overlap_and_nonoverlap():
    segs = [np.array([[0, 0], [20, 0], [20, 20], [0, 20]], np.float32),
            np.array([[30, 30], [40, 30], [40, 40], [30, 40]], np.float32)]
    masks, order = rasterize_segments(segs, 64, 64, mask_ratio=4, overlap=True)
    assert masks.shape == (1, 16, 16) and masks.dtype == np.uint8
    assert sorted(np.unique(masks).tolist()) == [0, 1, 2]
    assert order.tolist() == [0, 1]  # 面积降序：第一块更大 -> ID 1
    masks2, order2 = rasterize_segments(segs, 64, 64, mask_ratio=4, overlap=False)
    assert masks2.shape == (2, 16, 16) and order2 is None
    assert masks2[0].sum() > masks2[1].sum()
    empty, _ = rasterize_segments([], 64, 64, mask_ratio=4, overlap=True)
    assert empty.shape == (1, 16, 16) and empty.sum() == 0


def test_merge_multi_segment_matches_official():
    """多段合并逐点对拍官方 merge_multi_segment（含中间段的正/反向拼接）"""
    conv = pytest.importorskip("ultralytics.data.converter")
    parts = [np.array([[0, 0], [10, 0], [10, 10], [0, 10]], np.float32),
             np.array([[20, 0], [30, 0], [30, 10], [20, 10]], np.float32),
             np.array([[40, 0], [50, 0], [50, 10], [40, 10]], np.float32)]
    ref = np.concatenate(conv.merge_multi_segment([p.reshape(-1) for p in parts]), axis=0)
    assert np.array_equal(merge_polygons(parts), ref)


def test_rasterize_matches_official():
    """栅格化逐像素对拍官方 polygon2mask / polygons2masks / polygons2masks_overlap"""
    utils = pytest.importorskip("ultralytics.data.utils")
    segs = [np.array([[4.7, 4.2], [20.3, 4.9], [20.8, 20.1], [4.1, 20.7]], np.float32),
            np.array([[30.2, 30.7], [40.9, 30.1], [40.3, 40.8], [30.6, 40.2]], np.float32)]
    ours, order = rasterize_segments(segs, 64, 64, mask_ratio=4, overlap=True)
    ref, ref_order = utils.polygons2masks_overlap((64, 64), [s.reshape(-1) for s in segs],
                                                  downsample_ratio=4)
    assert np.array_equal(ours[0], ref)
    assert order.tolist() == ref_order.tolist()
    ours2, _ = rasterize_segments(segs, 64, 64, mask_ratio=4, overlap=False)
    ref2 = utils.polygons2masks((64, 64), [s.reshape(-1) for s in segs], color=1,
                                downsample_ratio=4)
    assert np.array_equal(ours2, ref2)


def test_collate_seg():
    imgs = [torch.zeros(3, SIZE, SIZE), torch.zeros(3, SIZE, SIZE)]
    labels = [np.array([[0, 1, 1, 5, 5]], np.float32), np.array([[1, 2, 2, 6, 6]], np.float32)]
    masks = [np.ones((1, 16, 16), np.int32), np.zeros((1, 16, 16), np.int32)]
    out = collate_fn(list(zip(imgs, labels, masks)))
    assert len(out) == 3
    assert out[0].shape == (2, 3, SIZE, SIZE)
    assert out[1].shape == (2, 6) and out[1][:, 0].tolist() == [0, 1]
    assert out[2].shape == (2, 16, 16)


def test_seg_val_masks_are_letterbox_aligned(tmp_path):
    desc, _ = _make_dataset(tmp_path, [_ann(1, 1, [_rect(8, 8, 40, 40)])])
    from data.build import build_eval_dataset
    ds = build_eval_dataset(load_dataset(desc), "train")
    assert isinstance(ds, CocoDataset)
    labels, masks = ds.seg_val(0, SIZE, 1.0, (0, 0), mask_ratio=4, overlap=False)
    assert labels.shape == (1, 5) and np.allclose(labels[0, 1:], [8, 8, 40, 40])
    assert masks.shape == (1, 16, 16) and masks.sum() > 0


def test_augment_segments_identity_and_flip():
    cfg = TrainConfig(imgsz=SIZE, mosaic=0.0, mixup=0.0, copy_paste=0.0, fliplr=0.0, flipud=0.0,
                      hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, bgr=0.0, aug_scale=0.0, translate=0.0,
                      degrees=0.0, shear=0.0, overlap_mask=True, mask_ratio=4)
    img = np.full((SIZE, SIZE, 3), 100, np.uint8)
    labels = np.array([[0, 8, 8, 40, 40]], np.float32)
    segs = [np.array([[8, 8], [40, 8], [40, 40], [8, 40]], np.float32)]
    out_img, out_lbs, masks = augment(img, labels, lambda i: (img, labels, segs),
                                      cfg, np.random.default_rng(0), 0, 1, segments=segs)
    assert out_img.shape == (3, SIZE, SIZE) and masks.shape == (1, 16, 16)
    assert np.allclose(out_lbs, labels) and masks.max() == 1
    _, _, flipped = flip_lr(img, labels, segments=segs)
    assert np.allclose(flipped[0][:, 0], [SIZE - 8, SIZE - 40, SIZE - 40, SIZE - 8])
    # 无分割调用保持 2 元组（检测路径兼容）
    out_img2, out_lbs2 = augment(img, labels, lambda i: (img, labels), cfg,
                                 np.random.default_rng(0), 0, 1)
    assert out_img2.shape == (3, SIZE, SIZE) and out_lbs2.shape == (1, 5)


def test_letterbox_train_segments():
    img = np.full((SIZE, SIZE * 2, 3), 100, np.uint8)
    labels = np.array([[0, 8, 8, 40, 40]], np.float32)
    segs = [np.array([[8, 8], [40, 8], [40, 40], [8, 40]], np.float32)]
    out_img, out_lbs, masks = letterbox_train(img, labels, SIZE, segments=segs, mask_ratio=4)
    assert out_img.shape == (3, SIZE, SIZE) and masks.shape == (1, 16, 16)
    assert out_lbs.shape == (1, 5)


def test_copy_paste_seg_flip():
    """官方 flip 模式：翻转当前图，按 p 比例把无遮挡实例贴回并追加标签/多边形"""
    from data.augment import copy_paste_seg
    img = np.zeros((64, 64, 3), np.uint8)
    img[:, :32] = (10, 20, 30)
    img[:, 32:] = (40, 50, 60)
    labels = np.array([[0, 4, 4, 20, 20]], np.float32)
    segs = [np.array([[4, 4], [20, 4], [20, 20], [4, 20]], np.float32)]
    cfg = TrainConfig(imgsz=64, copy_paste=1.0, copy_paste_mode="flip")
    out_img, out_labels, out_segs = copy_paste_seg(img, labels, segs, cfg, np.random.default_rng(0))
    assert out_labels.shape == (2, 5) and len(out_segs) == 2
    # 翻转实例落在右侧 [44,60]；源像素来自镜像图的左侧（10,20,30）
    assert out_img[10, 50].tolist() == [10, 20, 30]
    # p 很小时 round(p*N)=0 -> no-op
    cfg0 = TrainConfig(imgsz=64, copy_paste=0.1, copy_paste_mode="flip")
    _, l0, s0 = copy_paste_seg(img, labels, segs, cfg0, np.random.default_rng(0))
    assert l0.shape == (1, 5) and len(s0) == 1


def test_seg_training_rejects_yolo_format(tmp_path):
    root = tmp_path / "yolo"
    (root / "images" / "train").mkdir(parents=True)
    (root / "labels" / "train").mkdir(parents=True)
    desc = tmp_path / "yolo.yaml"
    desc.write_text(f"format: yolo\npath: {root}\nnames: [a]\n"
                    "train: {images: images/train}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="YOLO txt segmentation labels are not supported"):
        build_train_dataset(TrainConfig(imgsz=SIZE), load_dataset(desc), "train", task="segment")
