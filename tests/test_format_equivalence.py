"""双格式等价性：同一批图分别写成 COCO json 与 YOLO txt，两条加载链路必须给出相同结果

覆盖三层：训练标签（像素）、评估 GT（targets / gt_dict）、pycocotools 指标（GT 来源两种形态）。
"""

import json
from pathlib import Path

import cv2
import numpy as np

from config.datasets import load_dataset
from config.train_config import TrainConfig
from data.build import build_eval_dataset, build_train_dataset
from eval.coco_evaluator import CocoEvaluator
from utils.engines import Detections

SIZE = 64
# 3 图 2 类；框为像素 [x1, y1, x2, y2]（coco 写 bbox xywh；yolo 写归一化 cxcywh）
BOXES = {
    1: [(10, 10, 30, 30, 0), (40, 5, 60, 25, 1)],
    2: [(5, 40, 25, 60, 1)],
    3: [(0, 0, 20, 20, 0)],
}


def _make_pair(tmp_path):
    """同一批图写两份（coco/yolo）+ 两份描述符；返回 (coco 描述符, yolo 描述符)"""
    coco_root = tmp_path / "coco"
    (coco_root / "images" / "train").mkdir(parents=True)
    (coco_root / "annotations").mkdir(parents=True)
    yolo_root = tmp_path / "yolo"
    (yolo_root / "images" / "train").mkdir(parents=True)
    (yolo_root / "labels" / "train").mkdir(parents=True)

    images, anns = [], []
    aid = 0
    for iid, boxes in BOXES.items():
        name = f"{iid:06d}.jpg"
        img = np.zeros((SIZE, SIZE, 3), np.uint8)
        for x1, y1, x2, y2, _ in boxes:
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), -1)
        cv2.imwrite(str(coco_root / "images" / "train" / name), img)
        cv2.imwrite(str(yolo_root / "images" / "train" / name), img)
        images.append({"id": iid, "file_name": name, "width": SIZE, "height": SIZE})
        lines = []
        for x1, y1, x2, y2, cls in boxes:
            aid += 1
            anns.append({"id": aid, "image_id": iid, "category_id": cls + 1,  # coco id 1.. 对应 idx 0..
                         "bbox": [x1, y1, x2 - x1, y2 - y1], "area": (x2 - x1) * (y2 - y1), "iscrowd": 0})
            lines.append(f"{cls} {(x1 + x2) / 2 / SIZE:.6f} {(y1 + y2) / 2 / SIZE:.6f} "
                         f"{(x2 - x1) / SIZE:.6f} {(y2 - y1) / SIZE:.6f}")
        (yolo_root / "labels" / "train" / f"{iid:06d}.txt").write_text("\n".join(lines) + "\n")

    (coco_root / "annotations" / "instances_train.json").write_text(json.dumps(
        {"images": images, "annotations": anns,
         "categories": [{"id": 1, "name": "cat"}, {"id": 2, "name": "dog"}]}))
    coco_desc = tmp_path / "coco.yaml"
    coco_desc.write_text(f"format: coco\npath: {coco_root}\nnames: [cat, dog]\n"
                         "train: {images: images/train, ann: annotations/instances_train.json}\n"
                         "val: {images: images/train, ann: annotations/instances_train.json}\n",
                         encoding="utf-8")
    yolo_desc = tmp_path / "yolo.yaml"
    yolo_desc.write_text(f"format: yolo\npath: {yolo_root}\nnames: [cat, dog]\n"
                         "train: {images: images/train}\nval: {images: images/train}\n", encoding="utf-8")
    return load_dataset(str(coco_desc)), load_dataset(str(yolo_desc))


def test_train_labels_identical(tmp_path):
    """训练侧（augment=False）：逐图像素标签与类号一致"""
    coco_spec, yolo_spec = _make_pair(tmp_path)
    coco = build_train_dataset(TrainConfig(), coco_spec, "train", augment=False)
    yolo = build_train_dataset(TrainConfig(), yolo_spec, "train", augment=False)
    assert len(coco) == len(yolo) == 3
    assert (coco.n_instances, coco.n_backgrounds) == (yolo.n_instances, yolo.n_backgrounds) == (4, 0)
    for i in range(3):
        _, lb_coco = coco[i]
        _, lb_yolo = yolo[i]
        assert lb_coco.shape == lb_yolo.shape, f"img{i}: {lb_coco.shape} vs {lb_yolo.shape}"
        np.testing.assert_allclose(lb_coco, lb_yolo, atol=1e-3)
    print("  训练标签逐像素一致（4 框 / 0 背景）")


def test_eval_targets_and_gt_dict_identical(tmp_path):
    """评估侧：targets() 逐框一致；YOLO 现搭的 GT dict 与 COCO json 等价"""
    coco_spec, yolo_spec = _make_pair(tmp_path)
    coco = build_eval_dataset(coco_spec, "val")
    yolo = build_eval_dataset(yolo_spec, "val")

    for i in range(3):
        coco.load_image(i)
        yolo.load_image(i)
        a, b = coco.targets(i), yolo.targets(i)
        assert len(a) == len(b)
        for (box_a, cls_a), (box_b, cls_b) in zip(a, b):
            assert cls_a == cls_b
            np.testing.assert_allclose(box_a, box_b, atol=1e-3)

    gt = yolo.gt_dict()
    ref = json.loads((tmp_path / "coco" / "annotations" / "instances_train.json").read_text())
    assert [im["id"] for im in gt["images"]] == [im["id"] for im in ref["images"]]
    # YOLO 侧 category_id 恒为 0..nc-1（恒等映射），COCO 侧是原始 id：等价性看名字顺序
    assert [c["name"] for c in gt["categories"]] == [c["name"] for c in ref["categories"]]
    assert len(gt["annotations"]) == len(ref["annotations"])
    for a, b in zip(gt["annotations"], ref["annotations"]):
        assert (a["image_id"], a["category_id"] + 1, a["iscrowd"]) == (b["image_id"], b["category_id"], b["iscrowd"])
        np.testing.assert_allclose(a["bbox"], b["bbox"], atol=1e-3)
        np.testing.assert_allclose(a["area"], b["area"], atol=1.0)
    print("  targets 与 GT dict 双格式等价（4 框逐框一致）")


def test_evaluator_metrics_identical(tmp_path):
    """评估器两种 GT 来源（路径 vs 惰性 dict）在同一批完美检测上给出相同指标"""
    coco_spec, yolo_spec = _make_pair(tmp_path)
    coco = build_eval_dataset(coco_spec, "val")
    yolo = build_eval_dataset(yolo_spec, "val")

    ev_path = CocoEvaluator(coco.gt_source(), nc=len(coco.names))
    ev_lazy = CocoEvaluator(yolo.gt_source(), nc=len(yolo.names))

    for i in range(3):
        coco.load_image(i)
        yolo.load_image(i)
        boxes = np.array([b for b, _ in coco.targets(i)], np.float32)
        cls = np.array([c for _, c in coco.targets(i)], np.int64)
        dets = Detections(boxes, np.full(len(boxes), 0.99, np.float32), cls)
        ev_path.update(coco.image_id(i), dets)
        ev_lazy.update(yolo.image_id(i), dets)

    m_path, m_lazy = ev_path.compute(), ev_lazy.compute()
    for key in ("mAP@[.5:.95]", "mAP@50", "mAP@75", "AR@100", "P", "R", "images"):
        assert m_path[key] == m_lazy[key], f"{key}: {m_path[key]} vs {m_lazy[key]}"
    assert m_path["mAP@[.5:.95]"] == 1.0, f"完美检测应 mAP=1.0，实际 {m_path['mAP@[.5:.95]']}"
    assert [r[0] for r in m_path["per_class"]] == ["cat", "dog"] == [r[0] for r in m_lazy["per_class"]]
    print(f"  评估指标双格式一致：mAP50-95 {m_path['mAP@[.5:.95]']}（完美检测）")


def test_evaluator_lazy_requires_nc(tmp_path):
    """惰性 GT 缺 nc -> 明确报错（dict 形态的类别映射靠它）"""
    import pytest

    with pytest.raises(ValueError, match="`nc` is required"):
        CocoEvaluator(lambda: {"images": [], "categories": [], "annotations": []})
    print("  惰性 GT 缺 nc 报错正确")
