"""验收3：CocoEvaluator 在合成数据上指标正确（已知答案）"""

import json

import numpy as np

from eval.coco_evaluator import CocoEvaluator
from utils.engines import Detections


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
    # P/R 取 101 点召回网格上 F1 最大点：完美重合 -> 每类 1.0，all = 各类均值
    # （per-class 为原始浮点，插值数组带 1e-16 级噪声，故用容差）
    assert m["P"] == 1.0 and m["R"] == 1.0
    assert all(abs(row[3] - 1.0) < 1e-6 and abs(row[4] - 1.0) < 1e-6 for row in m["per_class"])


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
    assert m["P"] == 0.0 and m["R"] == 0.0


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


# ---- 分割（segm）口径 ----


def _seg_ann_file(tmp_path, h=64, w=64):
    """单图单类 GT，标注带 RLE segmentation"""
    from eval.coco_evaluator import encode_mask_rle

    mask = np.zeros((h, w), np.uint8)
    mask[10:30, 12:40] = 1
    rle = encode_mask_rle(mask)
    ann = {
        "images": [{"id": 1, "width": w, "height": h, "file_name": "x.jpg"}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 0, "iscrowd": 0,
                         "bbox": [12, 10, 28, 20], "area": float(mask.sum()), "segmentation": rle}],
        "categories": [{"id": 0, "name": "c0"}],
    }
    path = tmp_path / "instances_test.json"
    path.write_text(json.dumps(ann))
    return str(path), mask


def test_segm_perfect_match(tmp_path):
    """掩码与 GT 完全一致 -> mask mAP = 1.0（框 mAP 同时成立）"""
    gt, mask = _seg_ann_file(tmp_path)
    dets = Detections(boxes=np.array([[12, 10, 40, 30]], np.float32), scores=np.array([0.9], np.float32),
                      class_ids=np.array([0]), masks=mask[None])
    ev = CocoEvaluator(gt, nc=1, iou_type="segm")
    ev.update(1, dets)
    m = ev.compute()
    assert m["images"] == 1
    assert m["mAP@[.5:.95]"] == 1.0 and m["mAP@50"] == 1.0
    # 框与掩码打架时（框对、掩码空）mask mAP 必须掉下来
    empty = Detections(boxes=dets.boxes, scores=dets.scores, class_ids=dets.class_ids,
                       masks=np.zeros_like(dets.masks))
    ev2 = CocoEvaluator(gt, nc=1, iou_type="segm")
    ev2.update(1, empty)
    assert ev2.compute()["mAP@[.5:.95]"] == 0.0


def test_segm_requires_masks(tmp_path):
    """segm 口径下缺 masks 必须报错（不静默退化成框评估）"""
    gt, _ = _seg_ann_file(tmp_path)
    dets = Detections(boxes=np.array([[12, 10, 40, 30]], np.float32), scores=np.array([0.9], np.float32),
                      class_ids=np.array([0]))
    ev = CocoEvaluator(gt, nc=1, iou_type="segm")
    try:
        ev.update(1, dets)
    except ValueError as e:
        assert "masks" in str(e)
    else:
        raise AssertionError("segm evaluation must reject detections without masks")


def test_bbox_evaluator_ignores_masks(tmp_path):
    """bbox 口径不受 masks 影响（同一份 dets 可同时喂两个评估器）"""
    gt, mask = _seg_ann_file(tmp_path)
    dets = Detections(boxes=np.array([[12, 10, 40, 30]], np.float32), scores=np.array([0.9], np.float32),
                      class_ids=np.array([0]), masks=mask[None])
    m = CocoEvaluator(gt, nc=1, iou_type="bbox")
    m.update(1, dets)
    assert m.compute()["mAP@[.5:.95]"] == 1.0


def test_encode_mask_rle_roundtrip():
    """RLE 编码 -> pycocotools 解码 == 原掩码（含全零 / 全一 / 花式形状）"""
    from pycocotools import mask as M

    from eval.coco_evaluator import encode_mask_rle

    rng = np.random.default_rng(0)
    for mask in (np.zeros((7, 5), np.uint8),
                 np.ones((4, 6), np.uint8),
                 (rng.random((23, 31)) > 0.5).astype(np.uint8),
                 np.eye(16, dtype=np.uint8)):
        rle = encode_mask_rle(mask)
        assert isinstance(rle["counts"], str)  # results.json 要能直接序列化
        assert np.array_equal(M.decode(rle), mask)
        # 与官方编码器逐位一致（同一掩码同一 counts 字符串）
        assert rle["counts"] == M.encode(np.asfortranarray(mask))["counts"].decode("ascii")
