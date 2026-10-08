"""YOLO txt 数据集读取（训练/评估共用扫描；路径由调用方从描述符解析后传入）

格式约定（数据信息全部来自描述符 yaml，见 config/datasets/spec.py）：

- images: 图片目录（扫描 jpg/png/... 后缀）或 .txt 图片清单（每行一张；
  相对项相对清单文件所在目录解析，`#` 与空行跳过）
- labels: 与图片同名的 .txt，每行 `cls xc yc w h`（归一化 0-1，cls = 类别下标）。
  标签查找：`<stem>.txt` 优先；缺失时退到 `str(int(stem)).txt`（既兼容 1.txt <-> 000001.jpg
  的常见约定，也兼容 dataflow-cv coco2yolo 按 COCO image id 命名的输出）
- 无标签文件 / 空标签文件 = 背景图（YOLO 合法样本）；YOLO 无 crowd 概念（iscrowd 恒 0）

与 COCO 侧的分工：扫描期只存**归一化**标签；归一化 -> 像素必须有图尺寸，而尺寸要解码图片才知道，
所以像素换算推迟到 load_image(idx)（记录 (h, w)）之后由 targets() 完成——扫描期不逐张解码，
否则白多一趟全量 I/O。
"""

import math
import time
from pathlib import Path

import cv2
import numpy as np

from data.loader import SharedCounter, TrainMixin
from data.scan import ScanStats
from utils.progress import ProgressBar

__all__ = ["YoloDataset", "YoloTrainDataset", "list_images", "label_path_for", "parse_label_file",
           "scan_labels"]

IMG_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def list_images(src):
    """图片目录（按文件名排序）或 .txt 图片清单 -> [Path]"""
    src = Path(src)
    if src.suffix.lower() == ".txt":
        out = []
        for line in src.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            p = Path(line)
            out.append(p if p.is_absolute() else (src.parent / p))
        return out
    return sorted(p for p in src.iterdir() if p.suffix.lower() in IMG_SUFFIXES)


def label_path_for(img, labels_dir):
    """<stem>.txt 优先，缺失退 str(int(stem)).txt；都没有 -> None（= 背景图）"""
    labels_dir = Path(labels_dir)
    p = labels_dir / f"{Path(img).stem}.txt"
    if p.is_file():
        return p
    try:
        alt = labels_dir / f"{int(Path(img).stem)}.txt"
    except ValueError:
        return None
    return alt if alt.is_file() else None


def parse_label_file(path, nc):
    """标签 txt -> (M,5) float32 [cls, xc, yc, w, h]（归一化）；空文件 -> (0,5) 背景"""
    out = []
    for lineno, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            extra = " — segmentation masks are not supported" if len(parts) > 5 else ""
            raise ValueError(f"{path}:{lineno}: expected 5 values (cls xc yc w h), got {len(parts)}{extra}")
        cls, xc, yc, w, h = (float(v) for v in parts)
        if not all(math.isfinite(v) for v in (cls, xc, yc, w, h)):
            raise ValueError(f"{path}:{lineno}: non-finite value in '{line.strip()}'")
        if not cls.is_integer() or not 0 <= cls < nc:
            raise ValueError(f"{path}:{lineno}: class id {parts[0]} out of range [0, {nc})")
        if not (w > 0 and h > 0):
            raise ValueError(f"{path}:{lineno}: degenerate box w={w} h={h}")
        out.append([cls, xc, yc, w, h])
    return np.array(out, np.float32).reshape(-1, 5)


def _labels_to_pixels(lbs_norm, h, w):
    """归一化 (cls,xc,yc,w,h) -> 像素 (cls,x1,y1,x2,y2)（与 COCO 训练侧标签口径一致）"""
    if not len(lbs_norm):
        return np.zeros((0, 5), np.float32)
    out = np.empty_like(lbs_norm)
    out[:, 0] = lbs_norm[:, 0]
    out[:, 1] = (lbs_norm[:, 1] - lbs_norm[:, 3] / 2) * w
    out[:, 2] = (lbs_norm[:, 2] - lbs_norm[:, 4] / 2) * h
    out[:, 3] = (lbs_norm[:, 1] + lbs_norm[:, 3] / 2) * w
    out[:, 4] = (lbs_norm[:, 2] + lbs_norm[:, 4] / 2) * h
    return out


def scan_labels(src, images, labels_dir, nc, label="train", limit=0, progress=False, progress_file=None):
    """单趟扫描（对标 data/coco.py::scan_split）-> (kept_images, labels_norm, ScanStats)

    缺图从清单剔除并计数；无/空标签文件 = 背景。**全背景守卫**：扫到图但零个标签文件命中时
    直接报错点名 labels 目录（否则会"安静地训练 100% 背景"——标签目录填错是最常见的事故）。
    """
    prefix = f"{label}:".ljust(7)
    if progress:
        print(f"{prefix}Scanning {src} ({len(images)} images) · labels {labels_dir} ...", flush=True)
    images = images[:limit] if limit else images
    t0 = time.monotonic()
    bar = (ProgressBar(len(images), width=12, file=progress_file, pct=True, min_interval=0.25, start=t0)
           if progress else None)
    stride = max(1, len(images) // 400)
    kept, labels = [], []
    n_missing = n_bg = n_instances = n_hits = 0
    first_missing = None
    for i, img in enumerate(images, 1):
        if not img.is_file():
            n_missing += 1
            first_missing = first_missing or img.name
        else:
            lp = label_path_for(img, labels_dir)
            lbs = parse_label_file(lp, nc) if lp is not None else np.zeros((0, 5), np.float32)
            n_hits += lp is not None
            kept.append(img)
            labels.append(lbs)
            n_instances += len(lbs)
            n_bg += len(lbs) == 0
        if bar is not None and (i % stride == 0 or i == len(images)):
            speed = i / max(time.monotonic() - t0, 1e-6)
            bar.update(i, speed, desc=f"{prefix}Scanning {src}... {len(kept)} images, "
                                      f"{n_bg} backgrounds, {n_missing} missing:")
    if bar is not None:
        bar.close()
    if kept and n_hits == 0:
        tried = ", ".join(f"{Path(im).stem}.txt" for im in kept[:3])
        raise ValueError(f"no label files found under {labels_dir} for {len(kept)} images "
                         f"(tried e.g. {tried}) — wrong labels dir, or a different naming scheme")
    stats = ScanStats(n_missing, n_bg, 0, n_instances, 0.0, time.monotonic() - t0, first_missing)
    return kept, labels, stats


class YoloDataset:
    """按图片索引读取一个 YOLO 角色的图片与标签（评估口径）

    标签在扫描期为归一化形式；`targets()` 需要图尺寸 -> 必须先 `load_image(idx)`
    （评估循环/训练验证的调用序天然满足：先读图、再取 GT）。
    """

    def __init__(self, images_src, labels_dir, names, label="val", progress=False, progress_file=None):
        self.images_src = Path(images_src)
        self.labels_dir = Path(labels_dir)
        self.label = label
        self.names = dict(names)

        self.images, self.labels_norm, stats = scan_labels(
            self.images_src, list_images(self.images_src), self.labels_dir, len(self.names),
            label=label, progress=progress, progress_file=progress_file)
        self.n_instances = stats.n_instances
        self.n_categories = len(self.names)
        self.n_backgrounds = stats.n_backgrounds
        self.n_missing = stats.n_missing
        self.n_crowd_excluded = 0  # YOLO 无 crowd（字段对齐 COCO 侧，便于统一打印）
        self.parse_time = 0.0  # 无 json 解析段（scan_summary 据此省略 parse 项）
        self.scan_time = stats.scan_time
        self.first_missing = stats.first_missing
        self.n_corrupt = 0
        self.cat_id_to_idx = {i: i for i in self.names}
        self._dims = {}  # idx -> (h, w)，load_image 时记录

    def __len__(self):
        return len(self.images)

    def image_id(self, idx):
        """稳定图片 id（pycocotools 用；1-based）"""
        return idx + 1

    def load_image(self, idx):
        """(H, W, 3) BGR；读失败 -> 计数 + 空白图（GT 保留 → 记为漏检）"""
        img = cv2.imread(str(self.images[idx]))
        if img is None:
            self.n_corrupt += 1
            return np.zeros((1, 1, 3), np.uint8)
        self._dims[idx] = img.shape[:2]
        return img

    def _require_dims(self, idx):
        if idx not in self._dims:
            raise RuntimeError(f"normalized YOLO labels need the image size: call load_image({idx}) first")
        return self._dims[idx]

    def targets(self, idx, include_crowd=True):
        """第 idx 张图的 GT：[(xyxy, cls_idx), ...]（原图像素）

        include_crowd 接受但无意义（YOLO 无 crowd 概念，仅为与 COCO 侧接口一致）。
        """
        h, w = self._require_dims(idx)
        lbs = _labels_to_pixels(self.labels_norm[idx], h, w)
        return [([float(x1), float(y1), float(x2), float(y2)], int(c)) for c, x1, y1, x2, y2 in lbs]

    def gt_dict(self):
        """用已记录尺寸现搭 COCO GT dict（pycocotools 口径；未 load_image 过的图跳过）"""
        images, anns = [], []
        aid = 0
        for idx in range(len(self.images)):
            if idx not in self._dims:
                continue
            h, w = self._dims[idx]
            iid = self.image_id(idx)
            images.append({"id": iid, "file_name": self.images[idx].name, "width": w, "height": h})
            for (x1, y1, x2, y2), cls in self.targets(idx):
                aid += 1
                anns.append({"id": aid, "image_id": iid, "category_id": cls,
                             "bbox": [x1, y1, x2 - x1, y2 - y1],
                             "area": (x2 - x1) * (y2 - y1), "iscrowd": 0})
        return {"images": images,
                "categories": [{"id": i, "name": self.names[i]} for i in range(self.n_categories)],
                "annotations": anns}

    def gt_source(self):
        """pycocotools 的 GT 来源：惰性 callable（尺寸要到 load_image 之后才有）"""
        return self.gt_dict


class YoloTrainDataset(TrainMixin):
    """YOLO 训练集：load_image 返回 (BGR, 像素标签)，其余（增强/corrupt 计数）在 TrainMixin"""

    def __init__(self, cfg, images_src, labels_dir, names, label="train", augment=True, limit=0,
                 progress=False, progress_file=None):
        self.cfg = cfg
        self.use_augment = augment
        self.images_src = Path(images_src)
        self.labels_dir = Path(labels_dir)
        self.label = label
        self.names = dict(names)

        self.images, self.labels_norm, stats = scan_labels(
            self.images_src, list_images(self.images_src), self.labels_dir, len(self.names),
            label=label, limit=limit, progress=progress, progress_file=progress_file)
        self.n_instances = stats.n_instances
        self.n_categories = len(self.names)
        self.n_backgrounds = stats.n_backgrounds
        self.n_missing = stats.n_missing
        self.n_crowd_excluded = 0
        self.first_missing = stats.first_missing
        self.parse_time = 0.0
        self.scan_time = stats.scan_time
        self.cat_id_to_idx = {i: i for i in self.names}
        self._corrupt = SharedCounter()

    def __len__(self):
        return len(self.images)

    def image_id(self, idx):
        """稳定图片 id（调试/可视化用）"""
        return idx + 1

    def load_image(self, idx):
        """(BGR img, labels 像素副本)——mosaic/copy_paste/mixup 的邻图加载入口

        读不出来时计数 + 返回空白画布与空标签（当背景样本，不教假框），训练不因单张坏图中断。
        像素换算在这里完成（此刻有图尺寸）。
        """
        img = cv2.imread(str(self.images[idx]))
        if img is None:
            self._corrupt.bump()
            return np.zeros((self.cfg.imgsz, self.cfg.imgsz, 3), np.uint8), np.zeros((0, 5), np.float32)
        h, w = img.shape[:2]
        return img, _labels_to_pixels(self.labels_norm[idx], h, w)
