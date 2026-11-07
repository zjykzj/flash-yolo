"""LVIS 协议评估（faster_coco_eval lvis_style）在合成数据上的语义契约

覆盖三件 pycocotools 做不到的事：
1. neg_category_ids：图级"确认不存在"的类别出现预测 -> 记 FP
2. not_exhaustive_category_ids：图级标注不完整 -> 该类预测忽略
3. 既不在 GT、也不在 neg 列表的 (图, 类) 预测 -> 丢弃
另覆盖零检出图计入评估、空结果短路、per-class/APr/APc/APf 与 segm RLE 路径。
"""

import json

import numpy as np
import pytest

from eval.coco_evaluator import CocoEvaluator, encode_mask_rle
from eval.lvis_evaluator import LvisEvaluator
from utils.engines import Detections


def _lvis_ann(tmp_path, segment=False, h=64, w=64):
    """两张图 / 4 个类（r/c/f + 一个只在词汇里的类）

    img1: GT cat1(r), cat2(c); neg=[cat4]; not_exhaustive=[]
    img2: GT cat3(f);         neg=[cat1]; not_exhaustive=[cat2]
    cat4(f) 没有任何 GT —— 只作为 img1 的负样本，保证它不会进 per-class。
    """
    if segment:
        def mask(y, x, hh, ww):
            m = np.zeros((h, w), np.uint8)
            m[y:y + hh, x:x + ww] = 1
            return m

        m1, m2, m3 = mask(10, 10, 20, 20), mask(40, 40, 10, 10), mask(10, 10, 20, 20)
        seg1, seg2, seg3 = encode_mask_rle(m1), encode_mask_rle(m2), encode_mask_rle(m3)
    else:
        m1 = m2 = m3 = None
        seg1 = seg2 = seg3 = None

    def ann(aid, img, cat, box, seg):
        a = {"id": aid, "image_id": img, "category_id": cat, "bbox": box,
             "area": float(box[2] * box[3]), "iscrowd": 0}
        if seg is not None:
            a["segmentation"] = seg
        return a

    gt = {
        "info": {}, "licenses": [],
        "images": [
            {"id": 1, "file_name": "a.jpg", "width": w, "height": h,
             "neg_category_ids": [4], "not_exhaustive_category_ids": []},
            {"id": 2, "file_name": "b.jpg", "width": w, "height": h,
             "neg_category_ids": [1], "not_exhaustive_category_ids": [2]},
        ],
        "annotations": [
            ann(1, 1, 1, [10, 10, 20, 20], seg1),
            ann(2, 1, 2, [40, 40, 10, 10], seg2),
            ann(3, 2, 3, [10, 10, 20, 20], seg3),
        ],
        "categories": [
            {"id": 1, "name": "c0", "frequency": "r"},
            {"id": 2, "name": "c1", "frequency": "c"},
            {"id": 3, "name": "c2", "frequency": "f"},
            {"id": 4, "name": "c3", "frequency": "f"},
        ],
    }
    path = tmp_path / "lvis_test.json"
    path.write_text(json.dumps(gt))
    return str(path), (m1, m2, m3)


def _dets(boxes, scores, cls, masks=None):
    return Detections(boxes=np.asarray(boxes, np.float32).reshape(-1, 4),
                      scores=np.asarray(scores, np.float32).reshape(-1),
                      class_ids=np.asarray(cls, np.int64).reshape(-1),
                      masks=masks)


def _perfect(seg_masks=None):
    """img1: cat0 + cat1；img2: cat2（class index 0..2 = LVIS cat id 1..3）"""
    if seg_masks is None:
        d1 = _dets([[10, 10, 30, 30], [40, 40, 50, 50]], [0.9, 0.8], [0, 1])
        d2 = _dets([[10, 10, 30, 30]], [0.7], [2])
    else:
        m1, m2, m3 = seg_masks
        d1 = _dets([[10, 10, 30, 30], [40, 40, 50, 50]], [0.9, 0.8], [0, 1],
                   masks=np.stack([m1, m2]))
        d2 = _dets([[10, 10, 30, 30]], [0.7], [2], masks=m3[None])
    return [d1, d2]


def _run(ann_file, dets, iou_type="bbox", max_dets=300):
    ev = LvisEvaluator(ann_file, iou_type=iou_type, max_dets=max_dets)
    ev.update(1, dets[0])
    ev.update(2, dets[1])
    return ev.compute()


def test_perfect_match(tmp_path):
    ann_file, _ = _lvis_ann(tmp_path)
    m = _run(ann_file, _perfect())
    assert m["mAP@[.5:.95]"] == pytest.approx(1.0)
    assert m["mAP@50"] == pytest.approx(1.0)
    assert m["mAP@75"] == pytest.approx(1.0)
    assert m["APr"] == pytest.approx(1.0) and m["APc"] == pytest.approx(1.0) and m["APf"] == pytest.approx(1.0)
    assert m["AR@300"] == pytest.approx(1.0)
    assert m["images"] == 2
    assert len(m["per_class"]) == 3  # cat4 无 GT，不进 per-class


def test_negative_fp_penalized(tmp_path):
    """neg_category_ids 上的预测必须记 FP（该图确认没有这个类）"""
    ann_file, _ = _lvis_ann(tmp_path)
    dets = _perfect()
    # img2 的 neg=[cat1]，加一个 cat0 的高分 FP
    dets[1] = _dets([[10, 10, 30, 30], [0, 0, 5, 5]], [0.7, 0.99], [2, 0])
    m = _run(ann_file, dets)
    assert m["mAP@[.5:.95]"] < 1.0
    assert m["APr"] == pytest.approx(0.5)  # 高分 FP 在前，cat0 的 AP 减半


def test_not_exhaustive_ignored(tmp_path):
    """not_exhaustive_category_ids 上的预测被忽略（不记 FP）；COCO 口径会记 FP"""
    ann_file, _ = _lvis_ann(tmp_path)
    dets = _perfect()
    # img2 的 not_exhaustive=[cat2]，加一个 cat1 的高分预测
    dets[1] = _dets([[10, 10, 30, 30], [0, 0, 5, 5]], [0.7, 0.99], [2, 1])
    assert _run(ann_file, dets)["mAP@[.5:.95]"] == pytest.approx(1.0)

    coco = CocoEvaluator(ann_file)
    coco.update(1, dets[0])
    coco.update(2, dets[1])
    assert coco.compute()["mAP@[.5:.95]"] < 1.0  # COCO 口径把 not_exhaustive 预测记成 FP


def test_unlisted_category_dropped(tmp_path):
    """既不在 GT、也不在 neg 列表的 (图, 类) 预测被丢弃；COCO 口径会记 FP"""
    ann_file, _ = _lvis_ann(tmp_path)
    dets = _perfect()
    # cat3 在 img2 有 GT，但在 img1 既非 GT 也非 neg -> LVIS 丢弃、COCO 记 FP
    dets[0] = _dets([[10, 10, 30, 30], [40, 40, 50, 50], [0, 0, 5, 5]],
                    [0.9, 0.8, 0.99], [0, 1, 2])
    assert _run(ann_file, dets)["mAP@[.5:.95]"] == pytest.approx(1.0)

    coco = CocoEvaluator(ann_file)
    coco.update(1, dets[0])
    coco.update(2, dets[1])
    assert coco.compute()["mAP@[.5:.95]"] < 1.0  # COCO 口径把"未标注"当成无 GT -> FP


def test_zero_detection_image_counts_as_miss(tmp_path):
    """零检出图计入评估：img2 的 GT 必须记为漏检"""
    ann_file, _ = _lvis_ann(tmp_path)
    dets = _perfect()
    ev = LvisEvaluator(ann_file)
    ev.update(1, dets[0])
    ev.update(2, _dets([], [], []))
    m = ev.compute()
    assert m["images"] == 2
    assert m["mAP@[.5:.95]"] < 1.0


def test_no_prediction(tmp_path):
    ann_file, _ = _lvis_ann(tmp_path)
    ev = LvisEvaluator(ann_file)
    ev.update(1, _dets([], [], []))
    ev.update(2, _dets([], [], []))
    m = ev.compute()
    assert m["mAP@[.5:.95]"] == 0.0
    assert m["APr"] == 0.0 and m["APc"] == 0.0 and m["APf"] == 0.0
    assert m["per_class"] == []


def test_missing_frequency_raises(tmp_path):
    """lvis_style 的硬前提：categories 必须带 frequency"""
    ann = {
        "images": [{"id": 1, "file_name": "a.jpg", "width": 64, "height": 64}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 0, "bbox": [0, 0, 10, 10],
                         "area": 100, "iscrowd": 0}],
        "categories": [{"id": 0, "name": "c0"}],
    }
    path = tmp_path / "plain_coco.json"
    path.write_text(json.dumps(ann))
    with pytest.raises(ValueError, match="frequency"):
        LvisEvaluator(str(path))


def test_segm_perfect_match(tmp_path):
    """LVIS segm 路径：GT / 预测都走 RLE，完美匹配 -> mask AP = 1.0"""
    ann_file, masks = _lvis_ann(tmp_path, segment=True)
    m = _run(ann_file, _perfect(masks), iou_type="segm")
    assert m["mAP@[.5:.95]"] == pytest.approx(1.0)
    assert m["APr"] == pytest.approx(1.0) and m["APc"] == pytest.approx(1.0) and m["APf"] == pytest.approx(1.0)


def test_max_dets_option(tmp_path):
    """max_dets 可配置：与 Ultralytics 对齐时使用 100，返回键随之为 AR@100"""
    ann_file, _ = _lvis_ann(tmp_path)
    m = _run(ann_file, _perfect(), max_dets=100)
    assert "AR@100" in m and m["AR@100"] == pytest.approx(1.0)
