"""训练中快速 mAP 评估（COCO 口径近似，纯 numpy，无 pycocotools 依赖）

用途：训练中每轮验证的趋势指标与 best 选取（5000 张 ~秒级 vs pycocotools ~40-60s）。
正式数字以 scripts/eval.py（pycocotools 管线）为准——两套口径并存，见 CLAUDE.md。

口径近似（文档化）:
    - 贪婪匹配：每图每类按分数降序，IoU >= 阈值匹配未用 GT（同 COCOeval）
    - 101 点 PR 插值（同 COCOeval）；AP = 10 个 IoU 阈值（0.5:0.05:0.95）的均值；
      累积前按分数全局排序（同 COCOeval `_prepare`——跨图拼接序会 10 倍级低估）
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


def _greedy_match_all_thr(iou, det_scores):
    """单图单类贪婪匹配，全部 IoU 阈值一次遍历 -> tp (n_det, n_thr) bool

    iou: (n_det, n_gt) 与 det_scores 同序（均为分数降序）。每个阈值维护独立的
    已匹配集合，检测按分数降序依次取"未匹配 GT 中 IoU 最大者"，IoU >= 该阈值才匹配
    —— 与逐阈值分别遍历的旧实现逐位一致（同顺序、平局取小索引、已全匹配的阈值
    对后续检测只记 FP）。IoU 恒 >= 0，已匹配位置置 -1 即可让 argmax 只在未匹配
    集合上选；再用 ~matched 兜底，防止某阈值全匹配后 argmax 落到已匹配项。
    """
    n_det, n_gt = iou.shape
    n_thr = len(_IOU_THRESHOLDS)
    tp = np.zeros((n_det, n_thr), bool)
    if n_det == 0 or n_gt == 0:
        return tp
    # 整块 IoU 上限 < 最小阈值：任何阈值都不可能匹配（早期训练绝大多数检测属此类），
    # 直接全 FP 返回，跳过全部阈值循环
    if iou.max() < _IOU_THRESHOLDS[0]:
        return tp
    matched = np.zeros((n_thr, n_gt), bool)
    ti_idx = np.arange(n_thr)
    remaining = n_thr * n_gt  # Python 计数器：避免每迭代的 matched.all() 缩减（实测热点）
    for i in np.argsort(-det_scores):
        if remaining == 0:
            break
        row = np.where(matched, -1.0, iou[i])  # (n_thr, n_gt)
        j = row.argmax(1)  # (n_thr,) 各阈值选中的 GT
        ok = (iou[i][j] >= _IOU_THRESHOLDS) & (~matched[ti_idx, j])
        if ok.any():
            tp[i] = ok
            matched[ti_idx[ok], j[ok]] = True
            remaining -= int(ok.sum())
    return tp


def _ap_from_tp_fp(tp, fp, n_gt):
    """tp/fp 按分数降序累积 -> 101 点插值 AP（向量化：右侧运行最大值 + searchsorted）

    与逐点循环版逐位等价：recall 单调不减，`recall >= r` 的精度最大值 = 从首个
    recall>=r 位置起的右侧运行最大值；无命中点贡献 0。
    """
    if len(tp) == 0:
        return 0.0
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recall = tp_cum / max(n_gt, 1)
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1)
    prec_right_max = np.maximum.accumulate(precision[::-1])[::-1]
    idx = np.searchsorted(recall, _RECALL_POINTS, side="left")
    hit = idx < len(recall)
    pr = np.where(hit, prec_right_max[np.minimum(idx, len(recall) - 1)], 0.0)
    return float(pr.sum() / len(_RECALL_POINTS))


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

        # 每类 GT 计数：一次 bincount（替代 nc x n_images 的逐图扫描）
        gt_all = np.concatenate(self.gt_cls) if self.n_images else np.zeros(0, np.int64)
        n_gt_by_cls = np.bincount(gt_all, minlength=self.nc)

        ap50s, aps, ars = [], [], []
        prs, rcs = [], []  # IoU=0.5 下 max-F1 点的 P/R（每类）
        for c in range(self.nc):
            items = per_class[c]
            n_gt_c = int(n_gt_by_cls[c])
            if n_gt_c == 0:
                continue
            # 每 (图, 类) 的 IoU 矩阵只算一次（与阈值无关），10 个阈值复用；
            # 该类在本图无 GT 的组合只需记 FP（跳过 IoU 计算，早期训练/散类场景大头）
            prepared = []
            for img_idx, d_boxes, d_scores in items:
                m_gt = self.gt_cls[img_idx] == c
                prepared.append((d_scores, d_boxes, self.gt_boxes[img_idx][m_gt]))
            scores_all = np.concatenate([d_scores for d_scores, _, _ in prepared]) if prepared else np.zeros(0)
            # 阈值维度向量化：每 (图, 类) 的检测序列只遍历一次，再按阈值拆分
            per_thr_tp = [[] for _ in _IOU_THRESHOLDS]
            per_thr_fp = [[] for _ in _IOU_THRESHOLDS]
            for d_scores, d_boxes, gt_boxes in prepared:
                if len(gt_boxes):
                    tps = _greedy_match_all_thr(box_iou_matrix(d_boxes, gt_boxes), d_scores)
                else:
                    tps = np.zeros((len(d_scores), len(_IOU_THRESHOLDS)), bool)
                for ti in range(len(_IOU_THRESHOLDS)):
                    per_thr_tp[ti].append(tps[:, ti])
                    per_thr_fp[ti].append(~tps[:, ti])

            # COCOeval 口径：该类全部检测先按分数全局排序再累积 PR——逐图拼接顺序
            # 不等价（曾按图序直接累积，跨图高分/低分交错时 mAP 被低估 10 倍级，
            # 用同一批检测框对拍 pycocotools 定位；分数序与逐图匹配结果无关，仅影响累积）
            order = np.argsort(-scores_all, kind="stable")
            ap_by_thr = []
            rec_by_thr = []
            for ti, thr in enumerate(_IOU_THRESHOLDS):
                tp = np.concatenate(per_thr_tp[ti]) if per_thr_tp[ti] else np.zeros(0, bool)
                fp = np.concatenate(per_thr_fp[ti]) if per_thr_fp[ti] else np.zeros(0, bool)
                tp, fp = tp[order], fp[order]
                if ti == 0:  # IoU=0.5：全局按分数降序累积 -> max-F1 置信度点的 P/R
                    if len(tp):
                        tpc, fpc = np.cumsum(tp), np.cumsum(fp)
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
