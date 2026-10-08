"""为 YOLOv3-tiny 评估/重算锚点先验（darknet calc_anchors 口径 + YOLOv5 式覆盖判定）

用途：自定义数据集（或换了训练输入尺寸）**从零训练**前，判断现有锚组是否合适、要不要重算。
**官方 darknet 权重必须保留官方锚组**（权重与先验绑定，见 CLAUDE.md「YOLOv3-tiny」节）——
本脚本只面向从零训练。

判定口径（YOLOv5 `check_anchors` 同款）：对每个 GT 框取与锚组的最优形状比
    r = min_a max(w/a_w, a_w/w, h/a_h, a_h/h)
`r <= anchor_t(4.0)` 记为覆盖；覆盖率 >= 0.98 → 先验可接受（v5 即不重算）。注意 v3 的 exp 解码
理论上任何锚都能表示任意尺寸（没有 v5 那种硬覆盖上限），所以该覆盖率是**先验质量的经验标准**
（覆盖差 = 匹配/收敛变差），不是硬约束。

重算 = k-means：距离 = 1 - IoU（中心重合），像素单位 = 归一化宽高 × imgsz；
n_init 次随机初始化取平均 IoU 最优，seed 固定可复现；输出按面积升序、前一半给 P4/16（小）、
后一半给 P5/32（大），直接可粘进 config/models/yolov3-tiny.yaml 的 anchors 段。

用法:
    python scripts/compute_anchors.py --data <name|.yaml>              # 评估 + 重算 + 建议
    python scripts/compute_anchors.py --data coco --imgsz 640          # 指定训练输入尺寸
    python scripts/compute_anchors.py --data coco --anchors 23,27,37,58,81,82,81,82,135,169,344,319
    python scripts/compute_anchors.py --data coco --out anchors.yaml   # 片段落盘（默认只打印）
"""

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # 仓库根目录入 sys.path

import numpy as np
import yaml

from config.datasets import load_dataset
from config.train_config import TrainConfig
from data.build import build_train_dataset
from model.build import ARCHS
from utils.logger import get_logger, setup_logging

# 控制台立刻可用；文件日志走 logs/ 兜底（本脚本无 run 目录，与 convert_weights 同约定）
setup_logging()
logger = get_logger(__name__)

V3_CFG = ARCHS["yolov3-tiny"]["cfg"]


def iou_matrix(wh, anchors):
    """(N,2) 框宽高 vs (k,2) 锚宽高（中心重合）-> (N, k) IoU"""
    iw = np.minimum(wh[:, None, 0], anchors[None, :, 0])
    ih = np.minimum(wh[:, None, 1], anchors[None, :, 1])
    inter = iw * ih
    union = wh[:, None, 0] * wh[:, None, 1] + anchors[None, :, 0] * anchors[None, :, 1] - inter
    return inter / np.maximum(union, 1e-9)


def best_ratio(wh, anchors):
    """(N,) 每个 GT 与锚组的最优形状比 r = min_a max(w/aw, aw/w, h/ah, ah/h)（v5 覆盖口径）"""
    ratio = np.maximum.reduce([
        wh[:, None, 0] / anchors[None, :, 0], anchors[None, :, 0] / wh[:, None, 0],
        wh[:, None, 1] / anchors[None, :, 1], anchors[None, :, 1] / wh[:, None, 1],
    ])
    return ratio.min(1)


def anchor_stats(wh, anchors, anchor_t=4.0):
    """锚组在数据上的服务统计：best-IoU 分布 / 覆盖率 / 每锚服务占比与平均 IoU"""
    iou = iou_matrix(wh, anchors)
    best_iou, best_idx = iou.max(1), iou.argmax(1)
    per_anchor = []
    for j in range(len(anchors)):
        sel = best_idx == j
        per_anchor.append((int(sel.sum()) / len(wh), float(best_iou[sel].mean()) if sel.any() else 0.0))
    return {
        "iou_mean": float(best_iou.mean()),
        "iou_p10": float(np.percentile(best_iou, 10)),
        "coverage": float((best_ratio(wh, anchors) <= anchor_t).mean()),
        "per_anchor": per_anchor,
    }


def kmeans_anchors(wh, n, n_init=5, seed=0, iters=500):
    """k-means（距离 = 1 - IoU）-> (anchors (n,2) 面积升序, 最优聚类平均 IoU)

    n_init 次随机初始化（rng = seed + it），取平均 IoU 最高者；空簇保留上一轮中心。
    """
    wh32 = wh.astype(np.float32)
    best = (-1.0, None)
    for it in range(n_init):
        rng = np.random.default_rng(seed + it)
        centers = wh[rng.choice(len(wh), n, replace=False)].astype(np.float64)
        for _ in range(iters):
            assign = iou_matrix(wh, centers).argmax(1)
            onehot = np.eye(n, dtype=np.float32)[assign]  # (N, n)
            counts = onehot.sum(0)[:, None]
            new = (onehot.T @ wh32) / np.maximum(counts, 1)
            empty = (counts[:, 0] == 0)
            new[empty] = centers[empty]  # 空簇保位
            if np.allclose(new, centers, rtol=1e-5, atol=1e-4):
                centers = new
                break
            centers = new
        score = float(iou_matrix(wh, centers).max(1).mean())
        if score > best[0]:
            best = (score, centers.astype(np.float32))
    anchors = best[1][np.argsort(best[1][:, 0] * best[1][:, 1])]  # 面积升序
    return anchors, best[0]


def _fmt_anchors(anchors):
    half = len(anchors) // 2
    rows = []
    for i, (w, h) in enumerate(anchors):
        rows.append([round(float(w), 1), round(float(h), 1)])
    small = "[" + ", ".join(f"[{w}, {h}]" for w, h in rows[:half]) + "]"
    large = "[" + ", ".join(f"[{w}, {h}]" for w, h in rows[half:]) + "]"
    return f"anchors:\n  - {small}    # P4/16\n  - {large}    # P5/32"


def _load_anchors(path):
    """从模型 yaml 读 anchors（级序展平为 (k,2)）"""
    d = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    a = d.get("anchors")
    return None if not a else np.array([slot for lvl in a for slot in lvl], np.float32)


def collect_norm_wh(spec, role, limit, progress=True):
    """数据集的 GT 归一化宽高 (M,2)（COCO 用标注像素框 / 图像宽高；YOLO 直接读归一化标签，均不解码图片）"""
    ds = build_train_dataset(TrainConfig(), spec, role, augment=False, limit=limit, progress=progress)
    if spec.format == "coco":
        parts = []
        for img, lbs in zip(ds.images, ds.labels):
            if len(lbs):
                W, H = float(img["width"]), float(img["height"])
                parts.append(np.stack([(lbs[:, 3] - lbs[:, 1]) / W, (lbs[:, 4] - lbs[:, 2]) / H], 1))
        wh = np.concatenate(parts, 0) if parts else np.zeros((0, 2), np.float32)
    else:  # yolo：labels_norm (M,5) = [cls, xc, yc, w, h]（本就归一化）
        parts = [ds.labels_norm[i][:, 3:5] for i in range(len(ds)) if len(ds.labels_norm[i])]
        wh = np.concatenate(parts, 0) if parts else np.zeros((0, 2), np.float32)
    keep = (wh[:, 0] > 0) & (wh[:, 1] > 0)  # 退化框剔除（COCO 偶见零面积标注）
    return wh[keep].astype(np.float32), ds, int((~keep).sum())


def main():
    parser = argparse.ArgumentParser(description="evaluate / re-cluster yolov3-tiny anchor priors "
                                                 "(darknet calc_anchors convention)")
    parser.add_argument("--data", required=True,
                        help="dataset descriptor: a name in config/datasets/ (local/ wins) or a .yaml path")
    parser.add_argument("--role", default="train", help="descriptor role to cluster on (default train)")
    parser.add_argument("--imgsz", type=int, default=640, help="training input size the anchors are for")
    parser.add_argument("--n", type=int, default=6, help="number of anchors (even; half per level)")
    parser.add_argument("--n-init", type=int, default=5, help="k-means random restarts")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--anchor-t", type=float, default=4.0, help="shape-ratio coverage threshold (v5 anchor_t)")
    parser.add_argument("--anchors", default=None,
                        help="candidate anchor set to evaluate instead of the model yaml's, e.g. "
                             "'23,27,37,58,81,82,81,82,135,169,344,319' (w,h pairs, flat)")
    parser.add_argument("--limit", type=int, default=0, help="cluster on the first N images only (0 = all)")
    parser.add_argument("--out", default=None, help="also write the recommended anchors yaml fragment here")
    args = parser.parse_args()

    if args.n % 2 or args.n < 2:
        parser.error(f"--n must be even and >= 2 (half per detection level), got {args.n}")
    try:
        spec = load_dataset(args.data)
    except (ValueError, FileNotFoundError) as e:
        parser.error(str(e))

    wh_norm, ds, n_degenerate = collect_norm_wh(spec, args.role, args.limit)
    if len(wh_norm) < args.n:
        parser.error(f"only {len(wh_norm)} boxes found — need at least n={args.n} to cluster")
    wh = wh_norm * float(args.imgsz)  # 像素单位（锚的空间）
    logger.info(f"compute_anchors: {spec.source} ({spec.name}, role {args.role}) · {len(ds)} images · "
                f"{len(wh)} boxes ({n_degenerate} degenerate skipped) · imgsz {args.imgsz} · "
                f"seed {args.seed} · n {args.n}")

    if args.anchors:
        cur = np.array([float(v) for v in args.anchors.split(",")], np.float32).reshape(-1, 2)
    else:
        cur = _load_anchors(V3_CFG)
    if cur is not None:
        st = anchor_stats(wh, cur, args.anchor_t)
        served = " · ".join(f"[{i}] {100 * c:.1f}% iou {m:.3f}" for i, (c, m) in enumerate(st["per_anchor"]))
        logger.info(f"current : {len(cur)} anchors · bestIoU mean {st['iou_mean']:.3f} "
                    f"(p10 {st['iou_p10']:.3f}) · coverage(r<={args.anchor_t}) {100 * st['coverage']:.1f}%")
        logger.info(f"          per-anchor service: {served}")

    anchors, score = kmeans_anchors(wh, args.n, args.n_init, args.seed)
    st_new = anchor_stats(wh, anchors, args.anchor_t)
    logger.info(f"k-means : n_init {args.n_init} · centroid mean IoU {score:.3f} · "
                f"recommended-set coverage {100 * st_new['coverage']:.1f}% (bestIoU mean {st_new['iou_mean']:.3f})")

    if cur is not None:
        delta = st_new["iou_mean"] - st["iou_mean"]
        if st["coverage"] >= 0.98:
            logger.info(f"verdict : coverage {100 * st['coverage']:.1f}% >= 98% (v5 anchor_t {args.anchor_t}) — "
                        f"current anchor set fits this dataset; recompute not required")
        elif delta >= 0.03:
            logger.info(f"verdict : coverage {100 * st['coverage']:.1f}% < 98% and re-clustering clearly improves "
                        f"prior fit (bestIoU {st['iou_mean']:.3f} -> {st_new['iou_mean']:.3f}) — swap recommended "
                        f"(fragment below)")
        else:
            logger.info(f"verdict : coverage {100 * st['coverage']:.1f}% is below 98% (v5 rule), but re-clustering "
                        f"gives no material prior-fit gain (bestIoU {st['iou_mean']:.3f} -> {st_new['iou_mean']:.3f}) — "
                        f"either set is usable for v3's exp decode; swap optional")
        logger.info(f"          prior-fit: bestIoU mean {st['iou_mean']:.3f} (current) -> "
                    f"{st_new['iou_mean']:.3f} (re-clustered)")
    logger.info("recommended anchors (paste into config/models/yolov3-tiny.yaml):")
    logger.info(_fmt_anchors(anchors))
    if args.out:
        Path(args.out).write_text(_fmt_anchors(anchors) + "\n", encoding="utf-8")
        logger.info(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
