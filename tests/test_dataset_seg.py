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
from data.segments import (annotation_polygon, annotation_polygon_ex, decode_rle, merge_polygons,
                           rasterize_segments, rasterize_segments_multi, rle_polygon)
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


def _pycoco_rle(mask):
    """掩码 -> COCO RLE dict（压缩 counts；仅测试用 pycocotools 编码）"""
    mask_utils = pytest.importorskip("pycocotools.mask")
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode()
    return {"size": list(mask.shape), "counts": rle["counts"]}


def test_rle_decode_matches_pycocotools_and_uncompressed():
    """压缩字符串 / counts 列表两种 RLE 都与 cocoapi 解码逐像素一致"""
    pytest.importorskip("pycocotools.mask")
    h, w = 48, 64
    mask = np.zeros((h, w), np.uint8)
    cv2.rectangle(mask, (5, 7), (40, 30), 1, -1)
    cv2.circle(mask, (50, 35), 8, 1, -1)
    seg = _pycoco_rle(mask)
    assert np.array_equal(decode_rle(seg), mask)
    # 非压缩 counts 列表：4x4、第一列全 1（列优先计数 -> [0, 4, 12]）
    plain = {"size": [4, 4], "counts": [0, 4, 12]}
    expected = np.zeros((4, 4), np.uint8)
    expected[:, 0] = 1
    assert np.array_equal(decode_rle(plain), expected)
    # 非法输入明确报错（由 annotation_polygon_ex 回退）
    for bad in ({"size": [4, 4], "counts": [0, 3]}, {"size": [0, 4], "counts": []},
                {"size": [4, 4], "counts": "!!"}, {"counts": [0, 16]}):
        with pytest.raises((ValueError, TypeError)):
            decode_rle(bad)


def test_annotation_rle_polygon_fidelity():
    """RLE -> 轮廓多边形：单连通 IoU >= 0.99；多连通按官方 merge 口径 >= 0.95"""
    pytest.importorskip("pycocotools.mask")
    h, w = 64, 64
    cases = []
    rect = np.zeros((h, w), np.uint8)
    rect[8:40, 8:40] = 1
    cases.append(rect)
    circle = np.zeros((h, w), np.uint8)
    cv2.circle(circle, (32, 32), 18, 1, -1)
    cases.append(circle)
    multi = np.zeros((h, w), np.uint8)
    multi[4:20, 4:20] = 1
    multi[40:60, 40:60] = 1
    cases.append(multi)
    for mask in cases:
        ann = _ann(1, 1, _pycoco_rle(mask), bbox=[0, 0, w, h])
        poly, kind = annotation_polygon_ex(ann)
        assert kind == "rle" and poly is not None and len(poly) >= 3
        assert annotation_polygon(ann)[1] is False  # RLE 不算 fallback
        raster, _ = rasterize_segments([poly], h, w, mask_ratio=1, overlap=False)
        inter = np.logical_and(raster[0], mask).sum()
        union = np.logical_or(raster[0], mask).sum()
        assert inter / union >= (0.95 if mask is multi else 0.99)


def test_scan_decodes_rle_without_fallback(tmp_path):
    """扫描侧：有效 RLE 计入 n_seg_rle、不触发 bbox fallback，标签框由解码掩码导出"""
    pytest.importorskip("pycocotools.mask")
    mask = np.zeros((SIZE, SIZE), np.uint8)
    mask[8:40, 8:40] = 1
    desc, _ = _make_dataset(tmp_path, [_ann(1, 1, _pycoco_rle(mask), bbox=[8, 8, 32, 32])])
    ds = build_train_dataset(TrainConfig(imgsz=SIZE), load_dataset(desc), "train",
                             augment=False, task="segment")
    assert ds.n_seg_rle == 1 and ds.n_seg_fallback == 0
    assert len(ds.segments[0]) == 1
    img, labels, segments = ds.load_image(0)
    # 轮廓点是像素中心：框与标注 bbox 最多差 1px（掩码栅格化本身逐像素一致）
    assert labels.shape == (1, 5) and np.allclose(labels[0, 1:], [8, 8, 40, 40], atol=1.0)


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


def test_rasterize_multi_matches_single_calls():
    """多套输出与逐套调用逐像素一致（相同倍率共用 resize，overlap 各自排序）"""
    segs = [np.array([[4.7, 4.2], [40.3, 4.9], [40.8, 40.1], [4.1, 40.7]], np.float32),
            np.array([[20.2, 20.7], [30.9, 20.1], [30.3, 30.8], [20.6, 30.2]], np.float32)]
    ref_overlap = rasterize_segments(segs, 64, 64, mask_ratio=4, overlap=True)
    ref_plain = rasterize_segments(segs, 64, 64, mask_ratio=4, overlap=False)
    ref_r2 = rasterize_segments(segs, 64, 64, mask_ratio=2, overlap=False)
    out = rasterize_segments_multi(segs, 64, 64, [(4, True), (4, False), (2, False)])
    assert np.array_equal(out[0][0], ref_overlap[0]) and np.array_equal(out[0][1], ref_overlap[1])
    assert np.array_equal(out[1][0], ref_plain[0]) and out[1][1] is None
    assert np.array_equal(out[2][0], ref_r2[0]) and out[2][0].shape == (2, 32, 32)
    # 空实例：各 spec 的空掩码形状/类型与单套调用一致
    empty = rasterize_segments_multi([], 64, 64, [(4, True), (2, False)])
    assert np.array_equal(empty[0][0], rasterize_segments([], 64, 64, 4, True)[0])
    assert np.array_equal(empty[1][0], rasterize_segments([], 64, 64, 2, False)[0])


def test_seg_val_multi_matches_two_calls(tmp_path):
    """验证侧一次解析出 loss/指标两套 GT，与两次 seg_val 的结果逐像素一致"""
    desc, _ = _make_dataset(tmp_path, [
        _ann(1, 1, [_rect(20, 20, 36, 36)], bbox=[16, 16, 16, 16]),
        _ann(2, 2, [_rect(4, 4, 44, 44)], bbox=[4, 4, 40, 40]),
    ])
    from data.build import build_eval_dataset
    ds = build_eval_dataset(load_dataset(desc), "train")
    ref_loss = ds.seg_val(0, SIZE, 1.0, (0, 0), mask_ratio=2, overlap=True)
    ref_metric = ds.seg_val(0, SIZE, 1.0, (0, 0), mask_ratio=4, overlap=False)
    (lb_loss, m_loss), (lb_metric, m_metric) = ds.seg_val_multi(
        0, SIZE, 1.0, (0, 0), [(2, True), (4, False)])
    assert np.array_equal(lb_loss, ref_loss[0]) and np.array_equal(m_loss, ref_loss[1])
    assert np.array_equal(lb_metric, ref_metric[0]) and np.array_equal(m_metric, ref_metric[1])
    # overlap 索引图按面积降序 -> labels 同步重排（大实例在前；输入顺序是小、大）
    assert lb_loss[:, 0].tolist() == [1, 0] and m_loss.shape == (1, 32, 32)


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


def test_seg_training_accepts_yolo_bbox_fallback(tmp_path):
    """YOLO 5 列 bbox 行在分割训练下按矩形回退（不再拒绝整个格式）"""
    root = tmp_path / "yolo"
    (root / "images" / "train").mkdir(parents=True)
    (root / "labels" / "train").mkdir(parents=True)
    cv2.imwrite(str(root / "images" / "train" / "0001.jpg"), np.full((SIZE, SIZE, 3), 200, np.uint8))
    (root / "labels" / "train" / "0001.txt").write_text("0 0.5 0.5 0.5 0.5\n")
    desc = tmp_path / "yolo.yaml"
    desc.write_text(f"format: yolo\npath: {root}\nnames: [a]\n"
                    "train: {images: images/train}\n", encoding="utf-8")
    ds = build_train_dataset(TrainConfig(imgsz=SIZE), load_dataset(desc), "train",
                             augment=False, task="segment")
    assert ds.use_segments and ds.n_seg_fallback == 1
    _, lbs, segs = ds.load_image(0)
    assert lbs.shape == (1, 5) and np.allclose(lbs[0, 1:], [16, 16, 48, 48])
    assert segs[0].shape == (4, 2)
