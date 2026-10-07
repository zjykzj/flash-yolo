"""COCO 训练数据集 + collate

- ann json 一次解析为 per-image numpy 标签（~20MB），图像按需 cv2 加载
- 类别映射与 CocoDataset 共用 data/coco.py 的 parse_coco（category_id 排序 -> 0..nc-1）
- 训练侧丢弃 iscrowd=1 标注（crowd 只参与 COCO 评测的 AP50 扣分，不参与训练）
- 标签格式 (M, 5) [cls, x1, y1, x2, y2] 原图像素坐标；collate 后为 (N, 6) [batch_idx, ...]
- cfg 只按属性访问（duck-typing），data/ 不依赖 train/（单向依赖 train -> data）
"""

from pathlib import Path

import cv2
import numpy as np
import torch

from data.augment import augment, letterbox_train
from data.coco import parse_coco

__all__ = ["CocoTrainDataset", "collate_fn", "worker_init_fn"]

_worker_rng = None  # worker 进程内全局 RNG（worker_init_fn 播种，多进程下各 worker 独立）


class CocoTrainDataset:
    """COCO split 训练集（默认 train2017）"""

    def __init__(self, cfg, split="train2017", augment=True, limit=0):
        self.cfg = cfg  # TrainConfig 共享引用：close_mosaic 原地清零概率
        self.use_augment = augment

        ann_file = Path(cfg.data_dir) / "annotations" / f"instances_{split}.json"
        self.img_dir = Path(cfg.data_dir) / split
        if not self.img_dir.exists():
            self.img_dir = Path(cfg.data_dir) / "images" / split

        all_images, by_image, self.cat_id_to_idx, _ = parse_coco(ann_file)
        self.images = all_images[:limit] if limit else all_images

        # 官方 train2017 图片集比标注少 1 张（000000391895.jpg 缺失）——过滤文件不存在的图
        missing_ids = {img["id"] for img in self.images if not (self.img_dir / img["file_name"]).exists()}
        if missing_ids:
            print(f"[WARN] {split}: {len(missing_ids)} images missing on disk, excluded from training (e.g. id {sorted(missing_ids)[0]})")
            self.images = [img for img in self.images if img["id"] not in missing_ids]

        # 训练侧丢弃 iscrowd=1（val 侧 CocoDataset 保留）
        by_image = {img["id"]: [a for a in by_image.get(img["id"], []) if not a.get("iscrowd", 0)] for img in self.images}
        self.labels = []
        for img in self.images:
            lbs = [
                [self.cat_id_to_idx[a["category_id"]], *a["bbox"][:2], a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]]
                for a in by_image[img["id"]]
            ]
            self.labels.append(np.array(lbs, np.float32).reshape(-1, 5) if lbs else np.zeros((0, 5), np.float32))
        self.n_instances = sum(len(l) for l in self.labels)
        self.n_categories = len(self.cat_id_to_idx)

    def __len__(self):
        return len(self.images)

    def image_id(self, idx):
        """COCO image_id（调试/可视化用）"""
        return self.images[idx]["id"]

    def load_image(self, idx):
        """(BGR img, labels 副本)——mosaic/copy_paste/mixup 的邻图加载入口"""
        img = cv2.imread(str(self.img_dir / self.images[idx]["file_name"]))
        if img is None:
            raise FileNotFoundError(f"cannot read image: {self.img_dir / self.images[idx]['file_name']}")
        return img, self.labels[idx].copy()

    def __getitem__(self, idx):
        img, labels = self.load_image(idx)
        if self.use_augment:
            return augment(img, labels, self.load_image, self.cfg, _rng(), idx, len(self))
        return letterbox_train(img, labels, self.cfg.imgsz)

    def close_mosaic(self):
        """最后 N epoch：清零 mosaic/mixup/copy_paste 概率（HSV/flip/perspective 保留）"""
        self.cfg.mosaic = self.cfg.mixup = self.cfg.copy_paste = 0.0


def worker_init_fn(worker_id):
    """每 worker 独立播种 RNG（抽样流由 DataLoader 的 seed+epoch generator 保证可复现）

    并收紧 worker 内线程数：DataLoader 是多进程，若每个 worker 都用 OpenCV/torch 的默认
    线程池（= 核数），N 个 worker 会开出 N×核数 条线程互相抢占。实测（25 核，batch 64）：
    16 workers × 默认 25 线程 = 302 img/s；置 1 后 403 img/s（+34%）；20 workers × 1 = 496 img/s
    （+64%）。加载器是训练吞吐的瓶颈（GPU 利用率仅 ~14%），这条直接换来 epoch 提速。
    """
    global _worker_rng
    seed = int(torch.initial_seed() % (2**32))
    _worker_rng = np.random.default_rng(seed + worker_id)
    cv2.setNumThreads(1)
    torch.set_num_threads(1)


def _rng():
    """主进程用默认 RNG；worker 进程用 worker_init_fn 播种的 RNG"""
    global _worker_rng
    return _worker_rng if _worker_rng is not None else np.random.default_rng()


def collate_fn(batch):
    """-> (imgs (B,3,H,W) float32, targets (N,6) [batch_idx, cls, x1, y1, x2, y2])"""
    imgs, labels = zip(*batch)
    targets = []
    for bi, lbs in enumerate(labels):
        if len(lbs):
            targets.append(np.concatenate([np.full((len(lbs), 1), bi, np.float32), lbs], axis=1))
    tgts = torch.from_numpy(np.concatenate(targets)) if targets else torch.zeros((0, 6))
    return torch.stack(imgs), tgts
