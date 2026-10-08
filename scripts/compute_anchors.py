"""锚点先验计算（通用：任何 anchor-based 检测模型）+ YOLOv5 式覆盖判定

darknet calc_anchors 口径的通用实现：k-means 聚类数据集 GT 的（归一化宽高 × imgsz），
距离 = 1 - IoU（中心重合），n_init 次随机初始化取平均 IoU 最优，seed 固定可复现。
输出按面积升序**均分为 `--levels` 组**（级序 = 细 → 粗），与本仓库模型 yaml 的 anchors 段
schema 一致（每级一个 [w,h] 槽列表、级序与 Detect 的 from 顺序相同），直接可粘贴。
`--model`（默认 yolov3-tiny，当前唯一含锚模型）指定"现锚组"从哪个模型 yaml 读取并评估；
未来接入新的含锚模型：在其 yaml 写下 `anchors` 段即可复用本脚本，算法零改动。

判定口径（YOLOv5 `check_anchors` 同款，阈值可调）：每个 GT 与锚组的最优形状比
    r = min_a max(w/a_w, a_w/w, h/a_h, a_h/h)，r <= anchor_t 记为覆盖。
**阈值是解码相关的**：v5 的 4.0/98% 线对应其 (2σ)²∈(0,4) 的硬边界；darknet 系 exp 解码无硬
边界（理论任何锚可达任意尺寸），覆盖率只是先验质量的软信号——因此 verdict 同时看重聚类能把
bestIoU 提升多少（差 <0.03 判"可选/不必"）。

场景（见 CLAUDE.md「YOLOv3-tiny」节）：带官方权重 → 锚必须原样；从零训练且数据形状像 COCO
→ 实测官方锚收益≈0，不必重算；形状分布不同的自定义数据 → 跑本脚本看 verdict。

用法:
    python scripts/compute_anchors.py --data <name|.yaml>              # 默认评估 yolov3-tiny 现锚组
    python scripts/compute_anchors.py --data coco --imgsz 640          # 指定训练输入尺寸
    python scripts/compute_anchors.py --data coco --anchors 23,27,37,58,81,82,81,82,135,169,344,319
    python scripts/compute_anchors.py --data coco --n 9 --levels 3     # 3 级 × 3 槽（未来含锚模型）
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
from utils.logger import get_logger, log_params, setup_logging

# 控制台立刻可用；文件日志走 logs/ 兜底（本脚本无 run 目录，与 convert_weights 同约定）
setup_logging()
logger = get_logger(__name__)


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


def _fmt_anchors(anchors, levels):
    """锚组 -> 模型 yaml 片段（按面积升序均分为 levels 组，级序 = 细 → 粗）"""
    rows = [[round(float(w), 1), round(float(h), 1)] for w, h in anchors]
    per = len(rows) // levels
    out = ["anchors:"]
    for lvl in range(levels):
        chunk = rows[lvl * per:(lvl + 1) * per]
        tag = "level 1 (finest stride)" if lvl == 0 else f"level {lvl + 1}"
        out.append("  - [" + ", ".join(f"[{w}, {h}]" for w, h in chunk) + f"]    # {tag}")
    return "\n".join(out)


def _load_model_anchors(model_name):
    """模型 yaml 的 anchors 段 -> (现锚组 (k,2), 级数)；无 anchors 段返回 (None, None)"""
    d = yaml.safe_load(Path(ARCHS[model_name]["cfg"]).read_text(encoding="utf-8"))
    a = d.get("anchors")
    if not a:
        return None, None
    return np.concatenate([np.asarray(lvl, np.float32).reshape(-1, 2) for lvl in a], 0), len(a)


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
    parser = argparse.ArgumentParser(description="evaluate / re-cluster anchor priors "
                                                 "(darknet calc_anchors convention)")
    parser.add_argument("--data", required=True,
                        help="dataset descriptor: a name in config/datasets/ (local/ wins) or a .yaml path")
    parser.add_argument("--model", default="yolov3-tiny", choices=sorted(ARCHS),
                        help="model whose yaml `anchors:` section is evaluated as the current set")
    parser.add_argument("--role", default="train", help="descriptor role to cluster on (default train)")
    parser.add_argument("--imgsz", type=int, default=640, help="training input size the anchors are for")
    parser.add_argument("--levels", type=int, default=None,
                        help="output groups, finest -> coarsest (default: from the model yaml, else 2)")
    parser.add_argument("--n", type=int, default=None,
                        help="number of anchors to cluster (default: current set size, else 3 per level)")
    parser.add_argument("--n-init", type=int, default=5, help="k-means random restarts")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--anchor-t", type=float, default=4.0, help="shape-ratio coverage threshold (v5 anchor_t)")
    parser.add_argument("--anchors", default=None,
                        help="candidate anchor set to evaluate instead of the model yaml's, e.g. "
                             "'23,27,37,58,81,82,81,82,135,169,344,319' (w,h pairs, flat)")
    parser.add_argument("--limit", type=int, default=0, help="cluster on the first N images only (0 = all)")
    parser.add_argument("--out", default=None, help="also write the recommended anchors yaml fragment here")
    args = parser.parse_args()

    if args.anchors:
        cur = np.array([float(v) for v in args.anchors.split(",")], np.float32).reshape(-1, 2)
        cur_levels = None
    else:
        cur, cur_levels = _load_model_anchors(args.model)
    levels = args.levels or cur_levels or 2
    n = args.n or (len(cur) if cur is not None else 3 * levels)
    if n % levels or n < levels:
        parser.error(f"--n ({n}) must be divisible by --levels ({levels}) and >= levels")
    log_params(logger, __file__, data=args.data, role=args.role, model=args.model, imgsz=args.imgsz,
               n=n, levels=levels, **{"n-init": args.n_init}, seed=args.seed)
    try:
        spec = load_dataset(args.data)
    except (ValueError, FileNotFoundError) as e:
        parser.error(str(e))

    wh_norm, ds, n_degenerate = collect_norm_wh(spec, args.role, args.limit)
    if len(wh_norm) < n:
        parser.error(f"only {len(wh_norm)} boxes found — need at least n={n} to cluster")
    wh = wh_norm * float(args.imgsz)  # 像素单位（锚的空间）
    logger.info(f"compute_anchors: {spec.source} ({spec.name}, role {args.role}) · {len(ds)} images · "
                f"{len(wh)} boxes ({n_degenerate} degenerate skipped) · imgsz {args.imgsz} · "
                f"seed {args.seed} · n {n} ({levels} levels, model {args.model})")

    if cur is None:
        logger.info(f"current : no anchors in {Path(ARCHS[args.model]['cfg']).name} — evaluation skipped "
                    f"(pass `--anchors` to evaluate a candidate set instead)")
    else:
        st = anchor_stats(wh, cur, args.anchor_t)
        served = " · ".join(f"[{i}] {100 * c:.1f}% iou {m:.3f}" for i, (c, m) in enumerate(st["per_anchor"]))
        logger.info(f"current : {len(cur)} anchors · bestIoU mean {st['iou_mean']:.3f} "
                    f"(p10 {st['iou_p10']:.3f}) · coverage(r<={args.anchor_t}) {100 * st['coverage']:.1f}%")
        logger.info(f"          per-anchor service: {served}")

    anchors, score = kmeans_anchors(wh, n, args.n_init, args.seed)
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
    logger.info(f"recommended anchors (paste into {Path(ARCHS[args.model]['cfg']).name}'s anchors section):")
    logger.info(_fmt_anchors(anchors, levels))
    if args.out:
        Path(args.out).write_text(_fmt_anchors(anchors, levels) + "\n", encoding="utf-8")
        logger.info(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
