"""LVIS mAP 计算（faster_coco_eval，lvis_style）

与 `coco_evaluator.CocoEvaluator` 的分工：COCO 协议（maxDets=100、AP/AR 面积分组）走
pycocotools；LVIS 协议（`neg_category_ids` / `not_exhaustive_category_ids`、maxDets=300、
APr/APc/APf）走 faster_coco_eval 的 `lvis_style=True`。pycocotools 不认识 LVIS 的两个
负样本字段，直接用它会给出错误的 LVIS 数字（见 docs/zh/pipeline/eval.md）。

输入/输出契约与 CocoEvaluator 保持一致：
    update(image_id, detections)  # 累积一张图（类 id 是数据集索引 0..nc-1）
    compute() -> dict             # 指标 + per_class
唯一新增：APr / APc / APf 三个频率分组 AP，以及 `AR@<max_dets>` 键（默认 AR@300）。

实现注意：
- maxDets 必须显式设为 [300]（faster_coco_eval 默认 [1, 10, 100]，LVIS 官方口径是 300）。
- imgIds = 本次评估实际遍历的全部图片（含零检出图），否则漏检不计入 AP。
- lvis_style 要求 categories 带 frequency（r/c/f），缺失时报可操作的错误。
"""

import copy
import logging

import numpy as np

from eval.coco_evaluator import encode_mask_rle, pr_at_max_f1

__all__ = ["LvisEvaluator"]

logger = logging.getLogger(__name__)

_LVIS_MAX_DETS = 300
_AREA_ALL = 0


def _load_backend():
    """延迟导入 faster_coco_eval：COCO 评估不应因为缺包而失败，LVIS 路径给出可操作提示"""
    try:
        from faster_coco_eval import COCO, COCOeval_faster
    except ImportError as e:  # pragma: no cover - 环境缺依赖时给出可操作提示
        raise ImportError(
            "LVIS evaluation requires faster-coco-eval>=1.6.7; "
            "install it with `pip install faster-coco-eval`"
        ) from e
    return COCO, COCOeval_faster


class LvisEvaluator:
    """累积预测 -> faster_coco_eval(lvis_style=True)，compute() 输出 LVIS 指标

    Args:
        gt: GT 来源——ann json 路径 | COCO GT dict | 返回 GT dict 的 callable（惰性物化）
        nc: 类别数；callable 形态必填（dict 形态的 category_id 即 0..nc-1，建恒等映射）
        iou_type: "bbox"（框 IoU）| "segm"（掩码 IoU；update 需要 Detections.masks）
        max_dets: evaluator 侧 maxDets（LVIS 官方 300；与 Ultralytics 对比时可设 100）
    """

    def __init__(self, gt=None, nc=None, iou_type="bbox", max_dets=_LVIS_MAX_DETS):
        if iou_type not in ("bbox", "segm"):
            raise ValueError(f"iou_type must be 'bbox' or 'segm', got {iou_type!r}")
        if max_dets < 1:
            raise ValueError(f"max_dets must be >= 1, got {max_dets}")
        self.iou_type = iou_type
        self.max_dets = int(max_dets)
        self._gt_lazy = None
        if callable(gt):
            if nc is None:
                raise ValueError("LvisEvaluator: `nc` is required when `gt` is a callable (lazy GT source)")
            self.coco_gt = None
            self._gt_lazy = gt
            self.idx_to_cat_id = {i: i for i in range(nc)}  # dict 形态：category_id 恰为 0..nc-1
        else:
            self.coco_gt = self._make_coco(gt)
            self._validate_gt(self.coco_gt)
            cat_ids = sorted(self.coco_gt.getCatIds())
            self.idx_to_cat_id = {i: cid for i, cid in enumerate(cat_ids)}  # 0..nc-1 -> COCO category_id
        self.results = []
        self.image_ids = set()  # 全部评估图（含零检出）——compute 时作为 params.imgIds

    def _make_coco(self, gt):
        COCO, _ = _load_backend()
        return COCO(gt)

    @staticmethod
    def _validate_gt(coco_gt):
        """lvis_style 的硬前提：每个 category 带 frequency（r/c/f）"""
        cats = coco_gt.loadCats(coco_gt.getCatIds())
        missing = [c.get("name", c.get("id")) for c in cats if c.get("frequency") not in ("r", "c", "f")]
        if missing:
            raise ValueError(
                f"LVIS evaluation requires a `frequency` (r/c/f) on every category; "
                f"{len(missing)} missing, e.g. {missing[:5]} — use the official LVIS annotation "
                f"or set eval_style: coco for plain COCO data")

    def _materialize(self):
        """惰性 GT：首次需要时调用 callable 现搭"""
        if self.coco_gt is None:
            self.coco_gt = self._make_coco(self._gt_lazy())
            self._validate_gt(self.coco_gt)
            cat_ids = sorted(self.coco_gt.getCatIds())
            self.idx_to_cat_id = {i: cid for i, cid in enumerate(cat_ids)}
        return self.coco_gt

    def update(self, image_id, detections):
        """累积一张图的结果（COCO 结果格式：xywh + category_id + score [+ segmentation RLE]）

        即使 detections 为空也要记录 image_id：漏检图必须计入评估，否则 mAP 虚高。
        """
        self.image_ids.add(int(image_id))
        masks = getattr(detections, "masks", None)
        if self.iou_type == "segm" and masks is None:
            raise ValueError("segm evaluation requires Detections.masks (engine returned none)")
        for i, (box, score, cls_idx) in enumerate(zip(detections.boxes, detections.scores, detections.class_ids)):
            x1, y1, x2, y2 = box
            res = {
                "image_id": int(image_id),
                "category_id": self.idx_to_cat_id[int(cls_idx)],
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "score": float(score),
            }
            if self.iou_type == "segm":  # 只有 segm 评估器才付出 RLE 编码开销
                res["segmentation"] = encode_mask_rle(masks[i])
            self.results.append(res)

    def _empty_metrics(self):
        return {
            "mAP@[.5:.95]": 0.0,
            "mAP@50": 0.0,
            "mAP@75": 0.0,
            "mAP_small": 0.0,
            "mAP_medium": 0.0,
            "mAP_large": 0.0,
            f"AR@{self.max_dets}": 0.0,
            "APr": 0.0,
            "APc": 0.0,
            "APf": 0.0,
            "P": 0.0,
            "R": 0.0,
            "images": 0,
            "per_class": [],
        }

    @staticmethod
    def _metric(stats, key):
        """stats_as_dict 的 -1 / nan（该组合无有效评估）归一化为 0"""
        v = float(stats.get(key, -1.0))
        if not np.isfinite(v) or v < 0:
            return 0.0
        return round(v, 4)

    def _filter_results(self, coco_gt, img_ids):
        """只保留 LVIS 会评估的 (图, 类) 预测：GT ∪ neg_category_ids ∪ not_exhaustive_category_ids

        其余 pair 由 LVIS 规则直接丢弃；提前过滤有两个收益：
        1. 避免 faster_coco_eval 在 lvis_style 下对 dropped det 查 img_sizes 触发
           `unordered_map::at`（segm 实测）；
        2. 结果从"每图 × 1203 类"降回真实候选量级，loadRes/evaluate 显著更快。
        """
        allowed = {}
        for img_id in img_ids:
            img = coco_gt.imgs[img_id]
            allowed[img_id] = (set(img.get("neg_category_ids", []))
                               | set(img.get("not_exhaustive_category_ids", [])))
        for ann in coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=img_ids)):
            allowed[ann["image_id"]].add(ann["category_id"])
        return [r for r in self.results if r["category_id"] in allowed.get(r["image_id"], ())]

    def compute(self):
        """返回 LVIS 指标 dict（无匹配时 -1 归一化为 0；完全无预测时直接返回 0）"""
        if not self.results:
            return self._empty_metrics()
        coco_gt = self._materialize()
        # 评估范围 = 本次评估遍历的全部图片（含零检出图）；只有 update 从未调用时才退回有预测的图
        img_ids = sorted(self.image_ids) if self.image_ids else sorted({r["image_id"] for r in self.results})
        results = self._filter_results(coco_gt, img_ids)
        if not results:
            return self._empty_metrics()
        # faster_coco_eval.loadRes 会原地改写 segmentation 的 counts（bytes）；深拷贝隔离，
        # 保证 self.results 仍可 json.dump（scripts/eval.py 要落 results.json）
        coco_dt = coco_gt.loadRes(copy.deepcopy(results))
        _, COCOeval_faster = _load_backend()
        coco_eval = COCOeval_faster(coco_gt, coco_dt, iouType=self.iou_type, lvis_style=True,
                                    print_function=logger.debug)
        # LVIS 官方 maxDets=300；faster_coco_eval 默认 [1,10,100]，必须显式覆盖
        coco_eval.params.maxDets = [self.max_dets]
        coco_eval.params.imgIds = img_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        stats = coco_eval.stats_as_dict
        per_class = self._per_class(coco_eval, coco_gt)
        return {
            "mAP@[.5:.95]": self._metric(stats, "AP_all"),
            "mAP@50": self._metric(stats, "AP_50"),
            "mAP@75": self._metric(stats, "AP_75"),
            "mAP_small": self._metric(stats, "AP_small"),
            "mAP_medium": self._metric(stats, "AP_medium"),
            "mAP_large": self._metric(stats, "AP_large"),
            f"AR@{self.max_dets}": self._metric(stats, "AR_all"),
            "APr": self._metric(stats, "APr"),
            "APc": self._metric(stats, "APc"),
            "APf": self._metric(stats, "APf"),
            # P/R 无官方 stats 槽位：取各类 best-F1 点的均值（与 CocoEvaluator 同口径）
            "P": round(float(np.mean([r[3] for r in per_class])) if per_class else 0.0, 4),
            "R": round(float(np.mean([r[4] for r in per_class])) if per_class else 0.0, 4),
            "images": len(coco_eval.params.imgIds),
            "per_class": per_class,
        }

    def _per_class(self, coco_eval, coco_gt):
        """per-class 明细（仅含评估范围内有 GT 的类别）

        row = (类别名, 图片数, 实例数, P, R, AP50, AP@[.5:.95], AR)
        """
        eval_img_ids = set(coco_eval.params.imgIds)
        prec = coco_eval.eval["precision"]  # (T, R, K, A, M)
        rec_thrs = coco_eval.params.recThrs  # 101 点召回网格
        recall = coco_eval.eval["recall"]  # (T, K, A, M)
        maxdet = len(coco_eval.params.maxDets) - 1  # 当前只有一个 maxDets -> 索引 0
        rows = []
        for k, cat_id in self.idx_to_cat_id.items():
            img_ids = [i for i in coco_gt.getImgIds(catIds=cat_id) if i in eval_img_ids]
            if not img_ids:
                continue
            p50 = prec[0, :, k, _AREA_ALL, maxdet]
            p50 = p50[p50 > -1]
            ap50 = float(np.mean(p50)) if len(p50) else 0.0
            p = prec[:, :, k, _AREA_ALL, maxdet]
            p = p[p > -1]
            ap = float(np.mean(p)) if len(p) else 0.0
            r = recall[:, k, _AREA_ALL, maxdet]
            r = r[r > -1]
            ar = float(np.mean(r)) if len(r) else 0.0
            pr, rr = pr_at_max_f1(prec, rec_thrs, k, maxdet=maxdet)
            name = coco_gt.cats[cat_id]["name"]
            n_inst = len(coco_gt.getAnnIds(imgIds=img_ids, catIds=cat_id))
            rows.append((name, len(img_ids), n_inst, pr, rr, ap50, ap, ar))
        return rows
