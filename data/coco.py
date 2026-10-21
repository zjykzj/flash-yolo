"""COCO 数据集读取（训练/评估共用解析与扫描）

路径不再由本模块推导：调用方（data/build.py，源头是 config/datasets/spec.py 的描述符）
解出 ann_file 与 img_dir 后传入——此前 `annotations/instances_<split>.json` 与
`<root>/<split>|images/<split>` 的模板在 CocoDataset / CocoTrainDataset 两处各写了一遍。

解析口径（parse_coco）与扫描口径（scan_split）都在本模块，训练集与验证集共用，避免双份实现漂移。
"""

import json
import logging
import time
from pathlib import Path

import cv2
import numpy as np

from data.loader import SharedCounter, TrainMixin
from data.scan import ScanResult, ScanStats
from data.segments import (annotation_polygon, annotation_polygon_ex, polygon_bbox, rasterize_segments_multi,
                           transform_polygons, clip_polygons)
from utils.progress import ProgressBar

__all__ = ["CocoDataset", "CocoTrainDataset", "parse_coco", "scan_split"]

logger = logging.getLogger(__name__)


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


def scan_split(ann_file, img_dir, label, limit=0, drop_crowd=False, build_labels=False,
               build_segments=False, progress=False, progress_file=None):
    """解析 ann json + 单趟扫描 -> ScanResult

    exists 检查 / crowd 剔除 / 逐图标签构造合并进同一个循环：本机 118k 图实测扫描段 ≈2.6s，
    比原来三段各自遍历（1.0 + 0.44 + 5.6 ≈ 7.0s）快——省掉了 11.8 万条的 by_image 中间字典，
    同时能给出真实进度并顺手统计背景图 / 缺图 / crowd。

    训练侧 drop_crowd+build_labels（标签 (M,5) [cls,x1,y1,x2,y2] 原图像素坐标）；
    验证侧两者都不开（保留 crowd、不建标签，只计数——与官方 val 口径一致）。
    build_segments=True（分割训练）：标签框改由多边形包围盒导出，并保存合并后的多边形；
    RLE segmentation 解码为轮廓多边形（计入 n_seg_rle）；RLE 损坏/缺失/非法多边形才退回
    bbox 矩形并计入 n_seg_fallback（全部 fallback 直接报错）。
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
    segments = [] if build_segments else None
    n_missing = n_bg = n_crowd = n_instances = n_seg_fallback = n_seg_rle = 0
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
            if build_segments:
                lbs, img_segs, n_valid = [], [], 0
                for a in anns:
                    poly, kind = annotation_polygon_ex(a)
                    if poly is None:
                        continue
                    box = polygon_bbox(poly)
                    if box[2] <= box[0] or box[3] <= box[1]:
                        continue
                    if build_labels:
                        lbs.append([cat_id_to_idx[a["category_id"]], *box])
                    img_segs.append(poly)
                    n_seg_fallback += kind == "bbox"
                    n_seg_rle += kind == "rle"
                    n_valid += 1
                if build_labels:
                    labels.append(np.array(lbs, np.float32).reshape(-1, 5) if lbs else np.zeros((0, 5), np.float32))
                segments.append(img_segs)  # 与 kept/labels 逐图对齐（空图也占位）
                n_instances += n_valid
                n_bg += n_valid == 0
            else:
                if build_labels:
                    lbs = [[cat_id_to_idx[a["category_id"]], *a["bbox"][:2],
                            a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]] for a in anns]
                    labels.append(np.array(lbs, np.float32).reshape(-1, 5) if lbs else np.zeros((0, 5), np.float32))
                n_instances += len(anns)
                n_bg += not anns
            kept.append(img)
        if bar is not None and (i % stride == 0 or i == len(images)):
            speed = i / max(time.monotonic() - t_scan, 1e-6)
            # desc 只用文件名：完整路径已在上面的静态提示行；过长还会折行坏掉 \r 刷新
            bar.update(i, speed, desc=f"{prefix}Scanning {ann_file.name} {len(kept)} images, "
                                      f"{n_bg} backgrounds, {n_missing} missing:")
    if bar is not None:
        bar.close()  # 换行保留定格行（与训练/验证条同约定）
    if build_segments and n_instances and n_seg_fallback == n_instances:
        raise ValueError(
            f"segmentation training requires usable masks, but all {n_instances} annotations "
            f"in {ann_file} fell back to bbox rectangles (invalid RLE or missing segmentation) — "
            f"check the dataset descriptor / annotation format")
    stats = ScanStats(n_missing, n_bg, n_crowd, n_instances,
                      parse_time, time.monotonic() - t_scan, first_missing, n_seg_fallback, n_seg_rle)
    return ScanResult(kept, labels, by_image, cat_id_to_idx, names, stats, segments)


class CocoDataset:
    """按图片索引读取一个 COCO 角色的图片与标注（val/评估口径：保留 crowd、不建标签）"""

    def __init__(self, ann_file, img_dir, label="val", progress=False, progress_file=None):
        self.ann_file = Path(ann_file)
        self.img_dir = Path(img_dir)
        self.label = label

        # category_id（COCO 非连续 id）-> 0..nc-1（与训练侧共用 parse_coco 口径）
        res = scan_split(self.ann_file, self.img_dir, label, progress=progress, progress_file=progress_file)
        self.images, self.annotations = res.images, res.annotations
        self.cat_id_to_idx, self.names, stats = res.cat_id_to_idx, res.names, res.stats

        self.n_instances = stats.n_instances  # 含 crowd（与官方 val 口径一致）
        self.n_categories = len(self.names)
        self.n_backgrounds = stats.n_backgrounds
        self.n_missing = stats.n_missing
        self.n_crowd_excluded = 0  # val 侧不剔 crowd（字段对齐训练集，便于统一打印）
        self.n_seg_fallback = 0  # val 侧不做分割回退统计（只在分割训练扫描时累计）
        self.parse_time = stats.parse_time
        self.scan_time = stats.scan_time
        self.first_missing = stats.first_missing
        self.n_corrupt = 0  # 验证在父进程单进程内跑，普通计数即可（见 load_image）

    def gt_source(self):
        """pycocotools 的 GT 来源：COCO 直接给 ann json 路径（YOLO 侧返回 callable，见 data/yolo.py）"""
        return str(self.ann_file)

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

    def _seg_polys(self, idx, imgsz, ratio, pad, include_crowd):
        """第 idx 张图 -> (labels (M,5), polygons)；框由多边形包围盒导出（与训练侧同口径）

        polygon 已变换到 letterbox 画布 `imgsz×imgsz` 并裁剪；退化框（零宽/零高）连同
        多边形一起丢弃，保证 labels / polygons 行序一一对应。
        """
        info = self.images[idx]
        cls_ids, polys = [], []
        for a in self.annotations[info["id"]]:
            if not include_crowd and a.get("iscrowd", 0):
                continue
            poly, _fallback = annotation_polygon(a)
            if poly is None:
                continue
            cls_ids.append(self.cat_id_to_idx[a["category_id"]])
            polys.append(poly)
        if not polys:
            return np.zeros((0, 5), np.float32), []
        polys = clip_polygons(transform_polygons(polys, ratio, pad), imgsz, imgsz)
        labels, kept = [], []
        for c, poly in zip(cls_ids, polys):
            box = polygon_bbox(poly)
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            labels.append([c, *box])
            kept.append(poly)
        return np.array(labels, np.float32).reshape(-1, 5), kept

    def seg_val(self, idx, imgsz, ratio, pad, mask_ratio=4, overlap=False, include_crowd=False):
        """分割训练中验证：GT 多边形 -> letterbox 空间的 (labels (M,5), masks)

        labels 的框由多边形包围盒导出（与训练侧同口径）；polygon/框均落在 letterbox 画布
        `imgsz×imgsz`。overlap=True 时按栅格化面积降序重排 labels（与实例索引图一致）。
        训练中 Mask 指标始终用 overlap=False（保持标签顺序，框与掩码一一对应）。
        多项需求用 `seg_val_multi`（多边形只解析/栅格化一次）。
        """
        return self.seg_val_multi(idx, imgsz, ratio, pad, [(mask_ratio, overlap)],
                                  include_crowd=include_crowd)[0]

    def seg_val_multi(self, idx, imgsz, ratio, pad, specs, include_crowd=False):
        """一次解析/栅格化 -> 多套 (labels, masks)，顺序与 specs 一致

        specs: list[(mask_ratio, overlap)]。训练中验证同时需要 loss 网格
        （`mask_ratio`/`overlap_mask`）与指标网格（固定 4 / overlap=False），
        此前调用两次 `seg_val` 会重复逐实例 fillPoly；这里多边形只变换、填充一次。
        overlap=True 的 spec 按该网格的栅格化面积降序重排 labels。
        """
        lbs, polys = self._seg_polys(idx, imgsz, ratio, pad, include_crowd)
        out = []
        for masks, order in rasterize_segments_multi(polys, imgsz, imgsz, specs):
            out.append((lbs[order] if order is not None else lbs, masks))
        return out


class CocoTrainDataset(TrainMixin):
    """COCO 训练集：单趟扫描剔 crowd + 建标签（(M,5) [cls,x1,y1,x2,y2] 原图像素）

    增强入口 / close_mosaic / 惰性 corrupt 计数在 data/loader.TrainMixin（与 YOLO 训练集共用）。
    cfg 只按属性访问（duck-typing），data/ 不依赖 train/（单向依赖 train -> data）。
    """

    def __init__(self, cfg, ann_file, img_dir, label="train", augment=True, limit=0,
                 progress=False, progress_file=None, segment=False):
        self.cfg = cfg  # TrainConfig 共享引用：close_mosaic 原地清零概率
        self.use_augment = augment
        self.use_segments = segment
        self.ann_file = Path(ann_file)
        self.img_dir = Path(img_dir)
        self.label = label

        # 解析 + 单趟扫描（存在的图 / 剔 crowd / 建标签 / 计数在同一个循环里，见 data/coco.py::scan_split）
        res = scan_split(self.ann_file, self.img_dir, label, limit=limit, drop_crowd=True,
                         build_labels=True, build_segments=segment,
                         progress=progress, progress_file=progress_file)
        self.images, self.labels, self.cat_id_to_idx = res.images, res.labels, res.cat_id_to_idx
        self.segments = res.segments if segment else None
        self.names = res.names

        self.n_instances = res.stats.n_instances
        self.n_categories = len(self.cat_id_to_idx)
        self.n_backgrounds = res.stats.n_backgrounds
        self.n_missing = res.stats.n_missing
        self.n_crowd_excluded = res.stats.n_crowd_excluded
        self.n_seg_fallback = res.stats.n_seg_fallback
        self.n_seg_rle = res.stats.n_seg_rle
        if segment and self.n_seg_fallback:
            logger.warning(f"{self.n_seg_fallback} segmentation annotations had no usable polygon "
                           f"(invalid RLE or missing) — using bbox rectangles for those instances")
        self.first_missing = res.stats.first_missing
        self.parse_time, self.scan_time = res.stats.parse_time, res.stats.scan_time
        self._corrupt = SharedCounter()  # worker 里读失败 +1；父进程用 n_corrupt 读

    def __len__(self):
        return len(self.images)

    def image_id(self, idx):
        """COCO image_id（调试/可视化用）"""
        return self.images[idx]["id"]

    def load_image(self, idx):
        """(BGR img, labels 副本[, segments 副本])——mosaic/copy_paste/mixup 的邻图加载入口

        分割训练额外返回多边形列表（与 labels 行一一对应）。读不出来（文件损坏 / 训练中被删）时
        不再抛异常：计数 + 返回空白画布与空标签/空多边形（当背景样本，不教假框）。
        """
        img = cv2.imread(str(self.img_dir / self.images[idx]["file_name"]))
        if img is None:
            self._corrupt.bump()
            blank = np.zeros((self.cfg.imgsz, self.cfg.imgsz, 3), np.uint8)
            if self.use_segments:
                return blank, np.zeros((0, 5), np.float32), []
            return blank, np.zeros((0, 5), np.float32)
        if self.use_segments:
            return img, self.labels[idx].copy(), [p.copy() for p in self.segments[idx]]
        return img, self.labels[idx].copy()
