"""COCO 数据集读取（训练/评估共用解析与扫描）

目录约定（标准 COCO 布局）：
    <data_dir>/annotations/instances_<split>.json
    <data_dir>/<split>/<file_name>          （兼容 <data_dir>/images/<split>/）

解析口径（parse_coco）与扫描口径（scan_split）都在本模块，训练集与验证集共用，避免双份实现漂移。
"""

import json
import time
from collections import namedtuple
from pathlib import Path

import cv2
import numpy as np

from utils.progress import ProgressBar

__all__ = ["CocoDataset", "parse_coco", "scan_split", "scan_summary", "ScanResult", "ScanStats"]


def scan_summary(ds, extra=""):
    """扫描行下的 `└` 续行（训练与评估共用同一排版，避免两处漂移）

    形如 `       └ 849949 instances · 80 categories · crowd 10052 excluded · parse 12.6s + scan 6.6s`
    """
    crowd = f"crowd {ds.n_crowd_excluded} excluded · " if ds.n_crowd_excluded else ""
    return (f"       └ {ds.n_instances} instances · {ds.n_categories} categories · {crowd}"
            f"parse {ds.parse_time:.1f}s + scan {ds.scan_time:.1f}s{extra}")


def parse_coco(ann_file, status_line=None):
    """解析 COCO ann json -> (images, by_image, cat_id_to_idx, names)

    类别映射口径：category_id 排序 -> 0..nc-1（CocoDataset 与 CocoTrainDataset 共用，
    避免双份实现漂移）。

    status_line 非空时先打一行静态提示（如 `train: Scanning xxx/instances_train2017.json (448 MB) ...`）：
    json.load 是 C 层整体阻塞调用，中途无法刷新——实测解析期线程被 GIL 饿死（8s 只被调度 18 次，
    且全挤在最开始读文件的半秒内），所以只给"在读什么、多大"一行，进度交给紧随其后的扫描进度条。
    """
    if status_line:
        print(status_line, flush=True)  # 裸 print：这行要在 C 调用前落地，不能等 logger/缓冲
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


# 数据集扫描统计（训练/验证共用口径）
#   n_missing: 磁盘上不存在（已从 images 剔除）；n_backgrounds: 无框图片
#   n_crowd_excluded: 被剔除的 iscrowd=1 标注数（val 侧不剔除，恒 0）
ScanStats = namedtuple("ScanStats", "n_missing n_backgrounds n_crowd_excluded "
                                    "n_instances parse_time scan_time first_missing")

# 扫描结果：labels 仅训练侧非 None（build_labels=True）
ScanResult = namedtuple("ScanResult", "images labels annotations cat_id_to_idx names stats")


def scan_split(ann_file, img_dir, label, limit=0, drop_crowd=False, build_labels=False,
               progress=False, progress_file=None):
    """解析 ann json + 单趟扫描 -> ScanResult

    exists 检查 / crowd 剔除 / 逐图标签构造合并进同一个循环：本机 118k 图实测扫描段 ≈2.6s，
    比原来三段各自遍历（1.0 + 0.44 + 5.6 ≈ 7.0s）快——省掉了 11.8 万条的 by_image 中间字典，
    同时能给出真实进度并顺手统计背景图 / 缺图 / crowd。

    训练侧 drop_crowd+build_labels（标签 (M,5) [cls,x1,y1,x2,y2] 原图像素坐标）；
    验证侧两者都不开（保留 crowd、不建标签，只计数——与官方 val 口径一致）。
    缺图（官方 train2017 图片集比标注少 1 张的兜底）从 images 中剔除并计数。
    """
    prefix = f"{label}:".ljust(7)  # `train: ` / `val:   ` 对齐
    mb = ann_file.stat().st_size / 2**20  # MiB（与预览口径一致，非 SI 十进制 MB）
    t0 = time.monotonic()
    all_images, by_image, cat_id_to_idx, names = parse_coco(
        ann_file, status_line=f"{prefix}Scanning {ann_file} ({mb:.0f} MB) ..." if progress else None)
    parse_time = time.monotonic() - t0

    images = all_images[:limit] if limit else all_images
    # 时钟起点 = 解析开始：最终帧 elapsed 含解析，与 `└ ... parse Xs + scan Ys` 对得上
    bar = (ProgressBar(len(images), width=12, file=progress_file, pct=True, min_interval=0.25, start=t0)
           if progress else None)
    stride = max(1, len(images) // 400)  # 计数行的字符串构造也按 stride 走（11.8 万次 f-string 不值得）
    t_scan = time.monotonic()
    kept, labels = [], [] if build_labels else None
    n_missing = n_bg = n_crowd = n_instances = 0
    first_missing = None
    for i, img in enumerate(images, 1):
        anns = by_image.get(img["id"], [])
        if not (img_dir / img["file_name"]).exists():
            n_missing += 1
            first_missing = first_missing or img["file_name"]
        else:
            if drop_crowd:
                n_crowd += sum(1 for a in anns if a.get("iscrowd", 0))
                anns = [a for a in anns if not a.get("iscrowd", 0)]
            if build_labels:
                lbs = [[cat_id_to_idx[a["category_id"]], *a["bbox"][:2],
                        a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]] for a in anns]
                labels.append(np.array(lbs, np.float32).reshape(-1, 5) if lbs else np.zeros((0, 5), np.float32))
            kept.append(img)
            n_instances += len(anns)
            n_bg += not anns
        if bar is not None and (i % stride == 0 or i == len(images)):
            speed = i / max(time.monotonic() - t_scan, 1e-6)
            bar.update(i, speed, desc=f"{prefix}Scanning {ann_file}... {len(kept)} images, "
                                      f"{n_bg} backgrounds, {n_missing} missing:")
    if bar is not None:
        bar.close()  # 换行保留定格行（与训练/验证条同约定）
    stats = ScanStats(n_missing, n_bg, n_crowd, n_instances,
                      parse_time, time.monotonic() - t_scan, first_missing)
    return ScanResult(kept, labels, by_image, cat_id_to_idx, names, stats)


class CocoDataset:
    """按图片索引读取 COCO split 的图片与标注"""

    def __init__(self, data_dir, split="val2017", progress=False, progress_file=None):
        self.data_dir = Path(data_dir)
        self.split = split
        self.ann_file = self.data_dir / "annotations" / f"instances_{split}.json"

        self.img_dir = self.data_dir / split
        if not self.img_dir.exists():
            self.img_dir = self.data_dir / "images" / split

        # category_id（COCO 非连续 id）-> 0..nc-1（与训练侧共用 parse_coco 口径）
        # val 侧不剔 crowd、不建标签；progress=True 时打扫描行（默认关闭：eval/compare 输出不变）
        res = scan_split(self.ann_file, self.img_dir, "train" if split.startswith("train") else "val",
                         progress=progress, progress_file=progress_file)
        self.images, self.annotations = res.images, res.annotations
        self.cat_id_to_idx, self.names, stats = res.cat_id_to_idx, res.names, res.stats

        self.n_instances = stats.n_instances  # 含 crowd（与官方 val 口径一致）
        self.n_categories = len(self.names)
        self.n_backgrounds = stats.n_backgrounds
        self.n_missing = stats.n_missing
        self.n_crowd_excluded = 0  # val 侧不剔 crowd（字段对齐训练集，便于统一打印）
        self.parse_time = stats.parse_time
        self.scan_time = stats.scan_time
        self.first_missing = stats.first_missing
        self.n_corrupt = 0  # 验证在父进程单进程内跑，普通计数即可（见 load_image）

    def __len__(self):
        return len(self.images)

    def image_id(self, idx):
        """COCO image_id（评估器需要）"""
        return self.images[idx]["id"]

    def load_image(self, idx):
        """(H, W, 3) BGR

        读不出来时不再抛异常：计数 + 返回空白图（GT 保留 → 准确记为漏检），
        训练流程不因单张坏图中断。
        """
        info = self.images[idx]
        img = cv2.imread(str(self.img_dir / info["file_name"]))
        if img is None:
            self.n_corrupt += 1
            return np.zeros((1, 1, 3), np.uint8)  # letterbox 会把它补成灰底画布
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
