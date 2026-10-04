"""COCO 数据集读取（评估用）

目录约定（标准 COCO 布局）：
    <data_dir>/annotations/instances_<split>.json
    <data_dir>/<split>/<file_name>
"""

import json
from pathlib import Path

import cv2

__all__ = ["CocoDataset"]


class CocoDataset:
    """按图片索引读取 COCO split 的图片与标注"""

    def __init__(self, data_dir, split="val2017"):
        self.data_dir = Path(data_dir)
        self.split = split
        self.ann_file = self.data_dir / "annotations" / f"instances_{split}.json"
        with open(self.ann_file) as f:
            anns = json.load(f)

        self.img_dir = self.data_dir / split
        if not self.img_dir.exists():
            self.img_dir = self.data_dir / "images" / split

        # category_id（COCO 非连续 id）-> 0..nc-1
        categories = sorted(anns["categories"], key=lambda c: c["id"])
        self.cat_id_to_idx = {c["id"]: i for i, c in enumerate(categories)}
        self.names = {i: c["name"] for i, c in enumerate(categories)}

        self.images = anns["images"]
        by_image = {img["id"]: [] for img in self.images}
        for a in anns["annotations"]:
            if a["image_id"] in by_image:
                by_image[a["image_id"]].append(a)
        self.annotations = by_image

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

    def targets(self, idx):
        """第 idx 张图的 GT：[(xyxy, cls_idx), ...]（含 crowd 标注，与官方 val 口径一致）"""
        info = self.images[idx]
        out = []
        for a in self.annotations[info["id"]]:
            x, y, w, h = a["bbox"]
            out.append(([x, y, x + w, y + h], self.cat_id_to_idx[a["category_id"]]))
        return out
