"""TAL + STAL 标签分配（纯 torch，全矩阵口径，无截断近似）

TAL（Task Aligned Label Assignment）：
    对齐度 align = sigmoid(score)^alpha * CIoU^beta，per-GT 取 top-k 为正样本；
    多 GT 竞争的 anchor 只保留 CIoU 最高的 GT；软标签 = one-hot * 每 GT 归一化 align。
o2o 分支二次 top-k（topk_o2o 后按 CIoU 取 topk2）实现真一对一。

STAL（Small-Target-Aware Label Assignment）：
    仅候选筛选阶段，GT 短边 < s_min 时按中心扩到 s_ref 参与 anchor 中心包含测试；
    度量打分 / 正样本分配 / 回归目标一律使用原始框。

坐标口径：所有几何量在「逐 anchor 网格单位」下计算（GT / stride_i），与
utils/anchors.dist2bbox 的 ltrb 约定一致（x1 = ax - l, x2 = ax + r）。
"""

import torch
import torch.nn.functional as F

__all__ = ["TaskAlignedAssigner", "select_highest_overlaps", "bbox_iou_torch"]


def bbox_iou_torch(box1, box2, ciou=True, eps=1e-7):
    """IoU / CIoU（广播：box1 (..., 4) xyxy vs box2 (..., 4) xyxy）

    宽高 clamp 到 eps：预测框退化（ltrb -> 0，w=h=0）时 atan(w/h) = atan(0/0) = NaN
    （前向），反向 1/h = inf -> NaN 梯度（实测 o2o 分支静默死亡的根因）。
    """
    b1x1, b1y1, b1x2, b1y2 = box1[..., 0], box1[..., 1], box1[..., 2], box1[..., 3]
    b2x1, b2y1, b2x2, b2y2 = box2[..., 0], box2[..., 1], box2[..., 2], box2[..., 3]
    w1, h1 = (b1x2 - b1x1).clamp_min(eps), (b1y2 - b1y1).clamp_min(eps)
    w2, h2 = (b2x2 - b2x1).clamp_min(eps), (b2y2 - b2y1).clamp_min(eps)
    inter = (torch.min(b1x2, b2x2) - torch.max(b1x1, b2x1)).clamp(0) * (
        torch.min(b1y2, b2y2) - torch.max(b1y1, b2y1)
    ).clamp(0)
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union
    if not ciou:
        return iou
    cw = torch.max(b1x2, b2x2) - torch.min(b1x1, b2x1)
    ch = torch.max(b1y2, b2y2) - torch.min(b1y1, b2y1)
    c2 = cw**2 + ch**2 + eps
    rho2 = ((b2x1 + b2x2 - b1x1 - b1x2) ** 2 + (b2y1 + b2y2 - b1y1 - b1y2) ** 2) / 4
    v = (4 / torch.pi**2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)) ** 2
    with torch.no_grad():
        alpha = v / (v - iou + (1 + eps))
    return iou - (rho2 / c2 + v * alpha)


def select_highest_overlaps(mask_pos, iou, n_max_boxes=10):
    """多 GT 竞争的 anchor 只保留 CIoU 最高者

    Args:
        mask_pos: (N, M) bool 正样本标记
        iou: (N, M) CIoU（与 mask_pos 同形）
    返回: (N, M) bool
    """
    if mask_pos.shape[1] > n_max_boxes:
        # 每 anchor 最多只需考虑 CIoU 最高的 n_max 个 GT，其余不可能胜出
        topk_iou, _ = iou.topk(n_max_boxes, dim=1)
        mask_pos = mask_pos & (iou >= topk_iou[:, -1:])
    # 每 anchor 在其竞争 GT 中只保留 argmax
    max_iou, max_idx = iou.max(dim=1)  # (N,)
    one_max = F.one_hot(max_idx, iou.shape[1]).bool()  # (N, M)
    return mask_pos & one_max & (iou == max_iou.unsqueeze(1))


class TaskAlignedAssigner:
    """TAL + STAL（o2m: topk；o2o: topk_o2o + 二次 topk2 一对一）"""

    def __init__(self, nc=80, topk=10, topk_o2o=7, topk2=1, alpha=0.5, beta=6.0, s_min=8.0, s_ref=16.0):
        self.nc = nc
        self.topk = topk
        self.topk_o2o = topk_o2o
        self.topk2 = topk2
        self.alpha = alpha
        self.beta = beta
        self.s_min = s_min
        self.s_ref = s_ref

    def forward(self, pred_boxes, pred_scores, anchors, strides, gt, one2one=False):
        """分配一图目标

        Args:
            pred_boxes: (4, N) ltrb 距离（grid 单位）
            pred_scores: (nc, N) 分类 logits
            anchors: (2, N) anchor 中心（grid 单位）
            strides: (1, N) 逐点 stride
            gt: (M, 5) [cls, x1, y1, x2, y2] 像素坐标
            one2one: True 走 o2o 二次 top-k（真一对一）

        Returns:
            dict: fg_mask (N,) / target_scores (N, nc) 软标签 / target_ltrb (N, 4) /
                  target_boxes (N, 4) GT xyxy（逐 anchor grid 单位）/ n_pos
        """
        device = pred_boxes.device
        nc = pred_scores.shape[0]
        n = anchors.shape[1]
        out = {
            "fg_mask": torch.zeros(n, dtype=torch.bool, device=device),
            "target_scores": torch.zeros(n, nc, device=device),
            "target_ltrb": torch.zeros(n, 4, device=device),
            "target_boxes": torch.zeros(n, 4, device=device),
            "n_pos": 0,
        }
        m = gt.shape[0]
        if m == 0:
            return out

        gt_cls = gt[:, 0].long()
        # 逐 anchor 网格单位下的 GT（每 anchor 用自身 stride 归一）
        stride = strides[0]  # (N,)
        gt_grid = gt[:, 1:] / stride[:, None, None]  # (N, M, 4)

        # 预测框解码（逐 anchor 网格单位）: xyxy = [ax-l, ay-t, ax+r, ay+b]
        ax, ay = anchors[0], anchors[1]
        pred_xyxy = torch.stack([ax - pred_boxes[0], ay - pred_boxes[1], ax + pred_boxes[2], ay + pred_boxes[3]], 1)  # (N, 4)

        # STAL 候选筛选：短边 < s_min 的 GT 按中心扩到 s_ref（仅此测试用代理框）
        w, h = gt_grid[..., 2] - gt_grid[..., 0], gt_grid[..., 3] - gt_grid[..., 1]  # (N, M)
        s_min_g = self.s_min / stride[:, None]
        s_ref_g = self.s_ref / stride[:, None]
        widen_w, widen_h = w < s_min_g, h < s_min_g
        gx1 = torch.where(widen_w, (gt_grid[..., 0] + gt_grid[..., 2] - s_ref_g) / 2, gt_grid[..., 0])
        gy1 = torch.where(widen_h, (gt_grid[..., 1] + gt_grid[..., 3] - s_ref_g) / 2, gt_grid[..., 1])
        gx2 = torch.where(widen_w, (gt_grid[..., 0] + gt_grid[..., 2] + s_ref_g) / 2, gt_grid[..., 2])
        gy2 = torch.where(widen_h, (gt_grid[..., 1] + gt_grid[..., 3] + s_ref_g) / 2, gt_grid[..., 3])
        candidate = (ax[:, None] >= gx1) & (ax[:, None] <= gx2) & (ay[:, None] >= gy1) & (ay[:, None] <= gy2)  # (N, M)

        # 对齐度：sigmoid(cls_j score)^alpha * CIoU^beta（原始框）
        ciou = bbox_iou_torch(pred_xyxy[:, None, :], gt_grid)  # (N, M)
        score_gt_cls = pred_scores[gt_cls].T  # (N, M) 每 anchor 对 GT 类的分数
        # CIoU 可为负，浮点指数幂对负底数产生 NaN -> 先 clamp；sigmoid 项 clamp 为
        # 反向 s^(alpha-1) 在 s 下溢到 0 时防 inf（与上面 CIoU 的 0/0 修复互为双保险）
        align = score_gt_cls.sigmoid().clamp_min(1e-8).pow(self.alpha) * ciou.clamp_min(0).pow(self.beta)
        align = align * candidate

        # per-GT top-k 正样本
        k = self.topk_o2o if one2one else self.topk
        k = min(k, int(candidate.sum(0).max().item() if candidate.any() else 0))
        if k == 0:
            return out
        topk_align, _ = align.topk(k, dim=0)  # (k, M)
        mask_pos = align >= topk_align[-1].unsqueeze(0)  # (N, M) 含平局
        mask_pos = mask_pos & candidate

        if one2one:
            # 二次 top-k：per-GT 按 CIoU 只留 topk2（真一对一）
            ciou_masked = ciou * mask_pos
            top2, _ = ciou_masked.topk(self.topk2, dim=0)
            mask_pos = mask_pos & (ciou_masked >= top2[-1].unsqueeze(0))

        # 多 GT 竞争消解
        mask_pos = select_highest_overlaps(mask_pos, ciou)

        fg_mask = mask_pos.any(dim=1)  # (N,)
        if not fg_mask.any():
            return out
        gt_idx = mask_pos[fg_mask].float().argmax(dim=1)  # (n_pos,)

        # 软标签：per-GT 归一化 align
        align_sum = align.masked_fill(~mask_pos, 0).sum(0).clamp_min(1e-16)  # (M,)
        soft = align / align_sum.unsqueeze(0)  # (N, M)
        soft = soft * mask_pos
        target_scores = torch.zeros(n, nc, device=device)
        target_scores[fg_mask, gt_cls[gt_idx]] = soft[fg_mask][torch.arange(gt_idx.shape[0], device=device), gt_idx]

        # 回归目标（逐 anchor grid 单位，原始框）：l = ax-gx1, t = ay-gy1, r = gx2-ax, b = gy2-ay
        g = gt_grid[fg_mask, gt_idx]  # (n_pos, 4)
        target_ltrb = torch.zeros(n, 4, device=device)
        target_ltrb[fg_mask] = torch.stack(
            [ax[fg_mask] - g[:, 0], ay[fg_mask] - g[:, 1], g[:, 2] - ax[fg_mask], g[:, 3] - ay[fg_mask]], 1
        )
        target_boxes = torch.zeros(n, 4, device=device)
        target_boxes[fg_mask] = g

        return {
            "fg_mask": fg_mask,
            "target_scores": target_scores,
            "target_ltrb": target_ltrb,
            "target_boxes": target_boxes,
            "n_pos": int(fg_mask.sum()),
        }
