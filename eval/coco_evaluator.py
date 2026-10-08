"""COCO mAP 计算（pycocotools，COCO 评测事实标准）

选型说明：曾尝试纯 numpy 库 mean_average_precision，但其对小框（≤~19px）的匹配
完全失效（tests/test_evaluator.py 合成测试捕获），COCO 小目标（<32²）恰在其失效
区间，故弃用。pycocotools 是 COCO 官方评测实现，无此问题。

口径说明：官方 ultralytics 指标为自研实现（crowd 处理等细节略有差异），
本工程用 pycocotools 的结果与官方数字存在 ~0.1 量级的实现差异，属正常。

输出（compute() 返回 dict）：
    mAP@[.5:.95] / mAP@50 / mAP@75 / mAP_small / mAP_medium / mAP_large / AR@100
    P / R —— pycocotools 不直接给 P/R，这里取 **101 点召回网格上 F1 最大点**（= ultralytics
    "best-F1 置信度处取 P/R" 的插值近似，与训练侧 FastMetrics 同口径），"all" 行 = 各类均值
    per_class —— [(类别名, 图片数, 实例数, P, R, AP50, AP@[.5:.95], AR@100), ...]
"""

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

__all__ = ["CocoEvaluator"]

# COCOeval 数组索引：area 'all' = 0；maxDets=100 = 2（params.maxDets=[1,10,100]）
_AREA_ALL, _MAXDET_100 = 0, 2

_EMPTY_METRICS = {
    "mAP@[.5:.95]": 0.0,
    "mAP@50": 0.0,
    "mAP@75": 0.0,
    "mAP_small": 0.0,
    "mAP_medium": 0.0,
    "mAP_large": 0.0,
    "AR@100": 0.0,
    "P": 0.0,
    "R": 0.0,
    "images": 0,
    "per_class": [],
}


def pr_at_max_f1(prec, rec_thrs, k, iou=0, area=_AREA_ALL, maxdet=_MAXDET_100):
    """per-class P/R：101 点召回网格上 F1 最大点的 (P, R)

    Args:
        prec: COCOeval `eval["precision"]`，(T, 101, K, A, M)，插值后的查准率
        rec_thrs: 召回网格（`params.recThrs`，0..1 共 101 点）
        k: 类别下标
    """
    p = prec[iou, :, k, area, maxdet]
    keep = p > -1  # -1 = 该组合无有效评估（无 GT / IoU 过严）
    p, r = p[keep], rec_thrs[keep]
    if not len(p):
        return 0.0, 0.0
    f1 = np.where(p + r > 0, 2.0 * p * r / np.maximum(p + r, 1e-12), 0.0)
    i = int(np.argmax(f1))
    return float(p[i]), float(r[i])


def _make_coco(gt):
    """ann json 路径或 GT dict -> pycocotools COCO 对象

    COCO() 只接受路径（内部直接 open()），dict 形态手动装配（赋 dataset + 建索引）。
    """
    if isinstance(gt, dict):
        coco = COCO()
        coco.dataset = gt
        coco.createIndex()
        return coco
    return COCO(gt)


class CocoEvaluator:
    """累积预测 -> COCOeval，compute() 输出 COCO 指标与 per-class 明细

    Args:
        gt: GT 来源——ann json 路径 | COCO GT dict | 返回 GT dict 的 callable（惰性物化）
            callable 形态供 YOLO 数据集使用：归一化标签的像素化要等图尺寸，尺寸在评估循环里才拿到，
            因此 GT dict 只能在 compute()（循环结束后）现搭
        nc: 类别数；callable 形态必填（dict 形态的 category_id 即 0..nc-1，建恒等映射）
    """

    def __init__(self, gt=None, nc=None):
        self._gt_lazy = None
        if callable(gt):
            if nc is None:
                raise ValueError("CocoEvaluator: `nc` is required when `gt` is a callable (lazy GT source)")
            self.coco_gt = None
            self._gt_lazy = gt
            self.idx_to_cat_id = {i: i for i in range(nc)}  # dict 形态：category_id 恰为 0..nc-1
        else:
            self.coco_gt = _make_coco(gt)
            cat_ids = sorted(self.coco_gt.getCatIds())
            self.idx_to_cat_id = {i: cid for i, cid in enumerate(cat_ids)}  # 0..nc-1 -> COCO category_id
        self.results = []

    def _materialize(self):
        """惰性 GT：首次需要时调用 callable 现搭（compute 已在 redirect_prints 内，杂音不外泄）"""
        if self.coco_gt is None:
            self.coco_gt = _make_coco(self._gt_lazy())
        return self.coco_gt

    def update(self, image_id, detections):
        """累积一张图的结果（COCO 结果格式：xywh + category_id + score）"""
        for box, score, cls_idx in zip(detections.boxes, detections.scores, detections.class_ids):
            x1, y1, x2, y2 = box
            self.results.append(
                {
                    "image_id": int(image_id),
                    "category_id": self.idx_to_cat_id[int(cls_idx)],
                    "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    "score": float(score),
                }
            )

    def compute(self):
        """返回 COCO 指标 dict（无匹配时 -1 归一化为 0；完全无预测时 pycocotools 会崩溃，直接返回 0）"""
        if not self.results:
            return dict(_EMPTY_METRICS)
        coco_gt = self._materialize()
        coco_dt = coco_gt.loadRes(self.results)
        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        # 评估范围 = 有预测的图片集合（否则 --limit 子集评估会被全量 GT 稀释）
        coco_eval.params.imgIds = sorted({r["image_id"] for r in self.results})
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        s = np.nan_to_num(coco_eval.stats, nan=-1.0).clip(min=0.0)
        per_class = self._per_class(coco_eval, coco_gt)
        return {
            "mAP@[.5:.95]": round(float(s[0]), 4),
            "mAP@50": round(float(s[1]), 4),
            "mAP@75": round(float(s[2]), 4),
            "mAP_small": round(float(s[3]), 4),
            "mAP_medium": round(float(s[4]), 4),
            "mAP_large": round(float(s[5]), 4),
            "AR@100": round(float(s[8]), 4),  # stats[8] = AR @ maxDets=100
            # P/R 无官方 stats 槽位：取各类 best-F1 点的均值（ultralytics "all" 行同口径）
            "P": round(float(np.mean([r[3] for r in per_class])) if per_class else 0.0, 4),
            "R": round(float(np.mean([r[4] for r in per_class])) if per_class else 0.0, 4),
            "images": len(coco_eval.params.imgIds),
            "per_class": per_class,
        }

    def _per_class(self, coco_eval, coco_gt):
        """per-class 明细（仅含评估范围内有 GT 的类别）

        row = (类别名, 图片数, 实例数, P, R, AP50, AP@[.5:.95], AR@100)
        """
        eval_img_ids = set(coco_eval.params.imgIds)
        prec = coco_eval.eval["precision"]  # (T, R, K, A, M)
        rec_thrs = coco_eval.params.recThrs  # 101 点召回网格
        recall = coco_eval.eval["recall"]  # (T, K, A, M)
        rows = []
        for k, cat_id in self.idx_to_cat_id.items():
            img_ids = [i for i in coco_gt.getImgIds(catIds=cat_id) if i in eval_img_ids]
            if not img_ids:
                continue
            p50 = prec[0, :, k, _AREA_ALL, _MAXDET_100]
            p50 = p50[p50 > -1]
            ap50 = float(np.mean(p50)) if len(p50) else 0.0
            p = prec[:, :, k, _AREA_ALL, _MAXDET_100]
            p = p[p > -1]
            ap = float(np.mean(p)) if len(p) else 0.0
            r = recall[:, k, _AREA_ALL, _MAXDET_100]
            r = r[r > -1]
            ar = float(np.mean(r)) if len(r) else 0.0
            pr, rr = pr_at_max_f1(prec, rec_thrs, k)
            name = coco_gt.cats[cat_id]["name"]
            n_inst = len(coco_gt.getAnnIds(imgIds=img_ids, catIds=cat_id))
            rows.append((name, len(img_ids), n_inst, pr, rr, ap50, ap, ar))
        return rows
