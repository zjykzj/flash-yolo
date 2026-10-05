"""COCO 数据集读取（评估用）

目录约定（标准 COCO 布局）：
    <data_dir>/annotations/instances_<split>.json
    <data_dir>/<split>/<file_name>
"""

import json
from pathlib import Path

import cv2

__all__ = ["CocoDataset", "parse_coco"]


def parse_coco(ann_file):
    """解析 COCO ann json -> (images, by_image, cat_id_to_idx, names)

    类别映射口径：category_id 排序 -> 0..nc-1（CocoDataset 与 CocoTrainDataset 共用，
    避免双份实现漂移）。
    """
    with open(ann_file) as f:
        anns = json.load(f)
    categories = sorted(anns["categories"], key=lambda c: c["id"])
    cat_id_to_idx = {c["id"]: i for i, c in enumerate(categories)}
    names = {i: c["name"] for i, c in enumerate(categories)}
    by_image = {img["id"]: [] for img in anns["images"]}
    for a in anns["annotations"]:
        if a["image_id"] in by_image:
            by_image[a["image_id"]].append(a)
    return anns["images"], by_image, cat_id_to_idx, names


class CocoDataset:
    """按图片索引读取 COCO split 的图片与标注"""

    def __init__(self, data_dir, split="val2017"):
        self.data_dir = Path(data_dir)
        self.split = split
        self.ann_file = self.data_dir / "annotations" / f"instances_{split}.json"

        self.img_dir = self.data_dir / split
        if not self.img_dir.exists():
            self.img_dir = self.data_dir / "images" / split

        # category_id（COCO 非连续 id）-> 0..nc-1（与训练侧共用 parse_coco 口径）
        self.images, self.annotations, self.cat_id_to_idx, self.names = parse_coco(self.ann_file)
        self.n_instances = sum(len(v) for v in self.annotations.values())
        self.n_categories = len(self.names)

    def __len__(self):
        return len(self.images)

    def image_id(self, idx):
        """COCO image_id（评估器需要）"""
        return self.images[idx]["id"]

    def load_image(self, idx):
        """(H, W, 3) BGR"""
        info = self.images[idx]
        img = cv2.imread(str(self.img_dir / info["file_name"]))
        if img is None:
            raise FileNotFoundError(f"cannot read image: {self.img_dir / info['file_name']}")
        return img

    def targets(self, idx, include_crowd=True):
        """第 idx 张图的 GT：[(xyxy, cls_idx), ...]（默认含 crowd 标注，与官方 val 口径一致）

        include_crowd=False: 剔除 crowd（训练中快速指标用——crowd 不参与匹配）
        """
        info = self.images[idx]
        out = []
        for a in self.annotations[info["id"]]:
            if not include_crowd and a.get("iscrowd", 0):
                continue
            x, y, w, h = a["bbox"]
            out.append(([x, y, x + w, y + h], self.cat_id_to_idx[a["category_id"]]))
        return out
