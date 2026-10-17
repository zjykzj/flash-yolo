"""YOLO-seg txt：多边形解析、数据集契约、与 COCO json 路径的逐像素等价"""

import json

import cv2
import numpy as np
import pytest

from config.datasets import load_dataset
from config.train_config import TrainConfig
from data.build import build_eval_dataset, build_train_dataset
from data.yolo import parse_label_file_seg

SIZE = 64
NAMES = ["a", "b"]


def _write_image(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.full((SIZE, SIZE, 3), 200, np.uint8))


def _descriptor(tmp_path, root, fmt, split_dir="train"):
    if fmt == "yolo":
        body = (f"format: yolo\npath: {root}\nnames: [a, b]\n"
                f"train: {{images: images/{split_dir}, labels: labels/{split_dir}}}\n"
                f"val: {{images: images/{split_dir}, labels: labels/{split_dir}}}\n")
    else:
        body = (f"format: coco\npath: {root}\nnames: [a, b]\n"
                f"train: {{images: images/{split_dir}, ann: annotations/instances_train.json}}\n"
                f"val: {{images: images/{split_dir}, ann: annotations/instances_train.json}}\n")
    p = tmp_path / f"{fmt}.yaml"
    p.write_text(body, encoding="utf-8")
    return p


def _make_pair(tmp_path):
    """同一几何：多边形实例 + 无 segmentation 的 bbox 回退实例（COCO/YOLO 各一份）"""
    root = tmp_path / "data"
    _write_image(root / "images" / "train" / "0001.jpg")
    (root / "labels" / "train").mkdir(parents=True)
    # class0: 32x32 矩形多边形；class1: 12x12 bbox 行（回退）
    (root / "labels" / "train" / "0001.txt").write_text(
        "0 0.125 0.125 0.625 0.125 0.625 0.625 0.125 0.625\n"
        "1 0.78125 0.21875 0.1875 0.1875\n", encoding="utf-8")
    (root / "annotations").mkdir(parents=True)
    anns = [
        {"id": 1, "image_id": 1, "category_id": 1, "iscrowd": 0,
         "bbox": [8, 8, 32, 32], "area": 1024,
         "segmentation": [[8, 8, 40, 8, 40, 40, 8, 40]]},
        {"id": 2, "image_id": 1, "category_id": 2, "iscrowd": 0,
         "bbox": [44, 8, 12, 12], "area": 144, "segmentation": []},
    ]
    (root / "annotations" / "instances_train.json").write_text(json.dumps(
        {"images": [{"id": 1, "file_name": "0001.jpg", "width": SIZE, "height": SIZE}],
         "annotations": anns,
         "categories": [{"id": 1, "name": "a"}, {"id": 2, "name": "b"}]}))
    return _descriptor(tmp_path, root, "coco"), _descriptor(tmp_path, root, "yolo")


def test_parse_label_file_seg(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("0 0.1 0.1 0.5 0.1 0.5 0.5 0.1 0.5\n"
                 "1 0.7 0.3 0.2 0.1\n", encoding="utf-8")
    lbs, segs, nfb = parse_label_file_seg(p, 2)
    assert lbs.shape == (2, 5) and nfb == 1
    assert np.allclose(lbs[0], [0, 0.3, 0.3, 0.4, 0.4])
    assert np.allclose(lbs[1], [1, 0.7, 0.3, 0.2, 0.1])
    assert segs[0].shape == (4, 2) and segs[1] is None
    # 6 列（预测带 conf）/ 4 列 / 2 点多边形 / 类越界 / 退化多边形 / 退化框 -> 报错
    for bad in ("0 0.1 0.1 0.2 0.2 0.99\n", "0 0.1 0.1 0.2\n",
                "2 0.1 0.1 0.2 0.2 0.3 0.3\n",
                "0 0.2 0.2 0.2 0.2 0.2 0.2\n", "0 0 0 0 0\n"):
        p.write_text(bad, encoding="utf-8")
        with pytest.raises(ValueError):
            parse_label_file_seg(p, 2)


def test_yolo_seg_dataset_contract(tmp_path):
    coco_desc, yolo_desc = _make_pair(tmp_path)
    cfg = TrainConfig(imgsz=SIZE)
    spec = load_dataset(yolo_desc)
    ds = build_train_dataset(cfg, spec, "train", augment=False, task="segment")
    assert ds.use_segments and ds.n_seg_fallback == 1
    img, lbs, segs = ds.load_image(0)
    assert img.shape == (SIZE, SIZE, 3) and lbs.shape == (2, 5)
    assert np.allclose(lbs[0, 1:], [8, 8, 40, 40]) and np.allclose(lbs[1, 1:], [44, 8, 56, 20])
    assert segs[0].shape == (4, 2) and segs[1].shape == (4, 2)

    ev = build_eval_dataset(spec, "train", task="segment")
    ev.load_image(0)
    (lb_o, m_o), (lb_m, m_m) = ev.seg_val_multi(0, SIZE, 1.0, (0, 0), [(4, True), (4, False)])
    assert lb_o.shape == (2, 5) and m_o.shape == (1, 16, 16)
    assert lb_m.shape == (2, 5) and m_m.shape == (2, 16, 16)
    # overlap 索引图按面积降序：32x32 在前
    assert np.array_equal(lb_o[:, :2], lb_m[:, :2]) and m_o.sum() > 0
    gt = ev.gt_dict()
    assert len(gt["annotations"]) == 2
    assert all(a["segmentation"] and a["area"] > 0 for a in gt["annotations"])


def test_yolo_seg_matches_coco_path(tmp_path):
    """同几何：YOLO-seg txt 与 COCO json 的训练样本 / 验证掩码逐像素等价"""
    coco_desc, yolo_desc = _make_pair(tmp_path)
    cfg = TrainConfig(imgsz=SIZE)
    coco = build_train_dataset(cfg, load_dataset(coco_desc), "train", augment=False, task="segment")
    yolo = build_train_dataset(cfg, load_dataset(yolo_desc), "train", augment=False, task="segment")
    _, c_lbs, c_segs = coco.load_image(0)
    _, y_lbs, y_segs = yolo.load_image(0)
    assert np.allclose(c_lbs, y_lbs, atol=1e-5)
    assert len(c_segs) == len(y_segs)
    for a, b in zip(c_segs, y_segs):
        assert np.allclose(a, b, atol=1e-5)

    c_ev = build_eval_dataset(load_dataset(coco_desc), "train")
    y_ev = build_eval_dataset(load_dataset(yolo_desc), "train", task="segment")
    c_ev.load_image(0)
    y_ev.load_image(0)
    specs = [(4, True), (4, False)]
    c_out = c_ev.seg_val_multi(0, SIZE, 1.0, (0, 0), specs)
    y_out = y_ev.seg_val_multi(0, SIZE, 1.0, (0, 0), specs)
    for (c_lb, c_m), (y_lb, y_m) in zip(c_out, y_out):
        assert np.allclose(c_lb, y_lb, atol=1e-5)
        assert np.array_equal(c_m, y_m)
    print("  YOLO-seg vs COCO: labels/segments/masks 等价")
