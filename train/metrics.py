"""训练中快速 mAP 评估（COCO 口径近似，纯 numpy，无 pycocotools 依赖）

用途：训练中每轮验证的趋势指标与 best 选取（5000 张 ~秒级 vs pycocotools ~40-60s）。
正式数字以 scripts/eval.py（pycocotools 管线）为准——两套口径并存，见 CLAUDE.md。

口径近似（文档化）:
    - 贪婪匹配：每图每类按分数降序，IoU >= 阈值匹配未用 GT（同 COCOeval）
    - 101 点 PR 插值（同 COCOeval）；AP = 10 个 IoU 阈值（0.5:0.05:0.95）的均值
    - mAP = 各类 AP 的均值；AR@100 = 各阈值召回（maxDets=100）对类与阈值的均值
    - P/R = IoU 0.5 下 max-F1 置信度点的精度/召回（ultralytics 训练 val 同口径）
    - crowd 标注不参与（validator 传入 GT 时已剔除）
"""

import numpy as np

from utils.iou import box_iou_matrix

__all__ = ["FastMetrics"]

_IOU_THRESHOLDS = np.arange(0.5, 0.95 + 1e-9, 0.05)  # 10 个阈值
_RECALL_POINTS = np.linspace(0, 1, 101)  # 101 点插值
_MAX_DETS = 100  # 每图每类最多参与匹配的检测数（COCO maxDets=100 口径）


def _greedy_match(dets_xyxy, dets_scores, gt_xyxy, thr):
    """单图单类贪婪匹配 -> tp (n,) bool（分数降序，IoU>=thr 匹配未用 GT）"""
    order = np.argsort(-dets_scores)
    tp = np.zeros(len(dets_xyxy), bool)
    matched = np.zeros(len(gt_xyxy), bool)
    for i in order:
        if matched.all():
            break
        rem = np.where(~matched)[0]
        ious = box_iou_matrix(dets_xyxy[i : i + 1], gt_xyxy[rem])[0]
        j = int(ious.argmax())
        if ious[j] >= thr:
            tp[i] = True
            matched[rem[j]] = True
    return tp


def _ap_from_tp_fp(tp, fp, n_gt):
    """tp/fp 按分数降序累积 -> 101 点插值 AP"""
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recall = tp_cum / max(n_gt, 1)
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1)
    ap = 0.0
    for r in _RECALL_POINTS:
        mask = recall >= r
        if mask.any():
            ap += float(precision[mask].max())
    return ap / len(_RECALL_POINTS)


class FastMetrics:
    """累积每图 (检测, GT) -> compute() 输出 mAP@50 / mAP@[.5:.95] / AR@100 / images / instances"""

    def __init__(self, nc=80):
        self.nc = nc
        self.det_boxes, self.det_scores, self.det_cls = [], [], []
        self.gt_boxes, self.gt_cls = [], []
        self.n_images = 0
        self.n_instances = 0

    def update(self, det_boxes, det_scores, det_cls, gt_boxes, gt_cls):
        """累积一张图（均为原图像素坐标；gt 不含 crowd）"""
        self.n_images += 1
        self.det_boxes.append(np.asarray(det_boxes, np.float32).reshape(-1, 4))
        self.det_scores.append(np.asarray(det_scores, np.float32).reshape(-1))
        self.det_cls.append(np.asarray(det_cls, np.int64).reshape(-1))
        self.gt_boxes.append(np.asarray(gt_boxes, np.float32).reshape(-1, 4))
        self.gt_cls.append(np.asarray(gt_cls, np.int64).reshape(-1))
        self.n_instances += len(gt_cls)

    def compute(self):
        """COCO 口径近似指标"""
        if self.n_images == 0:
            return {"mAP@50": 0.0, "mAP@[.5:.95]": 0.0, "AR@100": 0.0, "P": 0.0, "R": 0.0,
                    "images": 0, "instances": 0}
        # 预分组：每类 -> [(img_idx, dets, dets_scores, gts)]（每图取 top-100 检测）
        per_class = {}
        for c in range(self.nc):
            per_class[c] = []
        for i in range(self.n_images):
            if not len(self.det_boxes[i]):
                continue
            for c in np.unique(self.det_cls[i]):
                m = self.det_cls[i] == c
                order = np.argsort(-self.det_scores[i][m])[:_MAX_DETS]
                per_class[int(c)].append((i, self.det_boxes[i][m][order], self.det_scores[i][m][order]))

        ap50s, aps, ars = [], [], []
        prs, rcs = [], []  # IoU=0.5 下 max-F1 点的 P/R（每类）
        for c in range(self.nc):
            items = per_class[c]
            n_gt_c = sum(len(self.gt_cls[i][self.gt_cls[i] == c]) for i in range(self.n_images))
            if n_gt_c == 0:
                continue
            ap_by_thr = []
            rec_by_thr = []
            for ti, thr in enumerate(_IOU_THRESHOLDS):
                tp_all, fp_all = [], []
                for img_idx, d_boxes, d_scores in items:
                    m_gt = self.gt_cls[img_idx] == c
                    tp = _greedy_match(d_boxes, d_scores, self.gt_boxes[img_idx][m_gt], thr)
                    tp_all.append(tp)
                    fp_all.append(~tp)
                tp = np.concatenate(tp_all) if tp_all else np.zeros(0, bool)
                fp = np.concatenate(fp_all) if fp_all else np.zeros(0, bool)
                if ti == 0:  # IoU=0.5：全局按分数降序累积 -> max-F1 置信度点的 P/R
                    if len(tp):
                        scores = np.concatenate([d_scores for _, _, d_scores in items])
                        order = np.argsort(-scores, kind="stable")
                        tpc, fpc = np.cumsum(tp[order]), np.cumsum(fp[order])
                        prec = tpc / np.maximum(tpc + fpc, 1)
                        rec = tpc / n_gt_c
                        f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
                        i = int(np.argmax(f1))
                        prs.append(float(prec[i]))
                        rcs.append(float(rec[i]))
                    else:
                        prs.append(0.0)
                        rcs.append(0.0)
                ap_by_thr.append(_ap_from_tp_fp(tp, fp, n_gt_c))
                rec_by_thr.append(tp.sum() / n_gt_c)
            ap50s.append(ap_by_thr[0])
            aps.append(float(np.mean(ap_by_thr)))
            ars.append(float(np.mean(rec_by_thr)))

        return {
            "mAP@50": round(float(np.mean(ap50s)), 4) if ap50s else 0.0,
            "mAP@[.5:.95]": round(float(np.mean(aps)), 4) if aps else 0.0,
            "AR@100": round(float(np.mean(ars)), 4) if ars else 0.0,
            "P": round(float(np.mean(prs)), 4) if prs else 0.0,
            "R": round(float(np.mean(rcs)), 4) if rcs else 0.0,
            "images": self.n_images,
            "instances": self.n_instances,
        }
