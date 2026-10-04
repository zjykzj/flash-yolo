"""COCO mAP 计算（pycocotools，COCO 评测事实标准）

选型说明：曾尝试纯 numpy 库 mean_average_precision，但其对小框（≤~19px）的匹配
完全失效（tests/test_evaluator.py 合成测试捕获），COCO 小目标（<32²）恰在其失效
区间，故弃用。pycocotools 是 COCO 官方评测实现，无此问题。

口径说明：官方 ultralytics 指标为自研实现（crowd 处理等细节略有差异），
本工程用 pycocotools 的结果与官方数字存在 ~0.1 量级的实现差异，属正常。
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
    "images": 0,
    "per_class": [],
}


class CocoEvaluator:
    """累积预测 -> COCOeval，compute() 输出 COCO 指标与 per-class 明细

    Args:
        ann_file: instances_<split>.json 路径
    """

    def __init__(self, ann_file):
        self.coco_gt = COCO(ann_file)
        cat_ids = sorted(self.coco_gt.getCatIds())
        self.idx_to_cat_id = {i: cid for i, cid in enumerate(cat_ids)}  # 0..nc-1 -> COCO category_id
        self.results = []

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
        coco_dt = self.coco_gt.loadRes(self.results)
        coco_eval = COCOeval(self.coco_gt, coco_dt, "bbox")
        # 评估范围 = 有预测的图片集合（否则 --limit 子集评估会被全量 GT 稀释）
        coco_eval.params.imgIds = sorted({r["image_id"] for r in self.results})
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        s = np.nan_to_num(coco_eval.stats, nan=-1.0).clip(min=0.0)
        return {
            "mAP@[.5:.95]": round(float(s[0]), 4),
            "mAP@50": round(float(s[1]), 4),
            "mAP@75": round(float(s[2]), 4),
            "mAP_small": round(float(s[3]), 4),
            "mAP_medium": round(float(s[4]), 4),
            "mAP_large": round(float(s[5]), 4),
            "AR@100": round(float(s[8]), 4),  # stats[8] = AR @ maxDets=100
            "images": len(coco_eval.params.imgIds),
            "per_class": self._per_class(coco_eval),
        }

    def _per_class(self, coco_eval):
        """per-class 明细：[(类别名, 图片数, 实例数, AP50, AP@[.5:.95], AR@100), ...]（仅含评估范围内有 GT 的类别）"""
        eval_img_ids = set(coco_eval.params.imgIds)
        prec = coco_eval.eval["precision"]  # (T, R, K, A, M)
        recall = coco_eval.eval["recall"]  # (T, K, A, M)
        rows = []
        for k, cat_id in self.idx_to_cat_id.items():
            img_ids = [i for i in self.coco_gt.getImgIds(catIds=cat_id) if i in eval_img_ids]
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
            name = self.coco_gt.cats[cat_id]["name"]
            n_inst = len(self.coco_gt.getAnnIds(imgIds=img_ids, catIds=cat_id))
            rows.append((name, len(img_ids), n_inst, ap50, ap, ar))
        return rows
