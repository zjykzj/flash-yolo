"""TAL + STAL 标签分配（纯 torch，批量 (B,N,M) 掩码口径，无截断近似）

TAL（Task Aligned Label Assignment）：
    对齐度 align = sigmoid(score)^alpha * CIoU^beta，per-GT 取 top-k 为正样本；
    多 GT 竞争的 anchor 只保留 CIoU 最高的 GT；软标签 = one-hot * 每 GT 归一化 align。
o2o 分支二次 top-k（topk_o2o 后按 CIoU 取 topk2）实现真一对一。

STAL（Small-Target-Aware Label Assignment）：
    仅候选筛选阶段，GT 短边 < s_min 时按中心扩到 s_ref 参与 anchor 中心包含测试；
    度量打分 / 正样本分配 / 回归目标一律使用原始框。

坐标口径：所有几何量在「逐 anchor 网格单位」下计算（GT / stride_i），与
utils/anchors.dist2bbox 的 ltrb 约定一致（x1 = ax - l, x2 = ax + r）。

批量口径（对齐官方语义）：整个 batch 的图一次算完（(B, N, M) 掩码张量，短图右侧
padding GT 由 gt_mask 屏蔽）；按元素预算分块控显存，块内再裁剪 padding 列。
声明 @torch.no_grad() 且调用方 detach 输入 —— 标签分配是离散操作，梯度不经过它
（官方 TaskAlignedAssigner 同口径）。单图 forward 是同一实现的薄包装。
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
    """多 GT 竞争的 anchor 只保留 CIoU 最高者（支持 (..., N, M) 任意前导维）

    Args:
        mask_pos: (..., N, M) bool 正样本标记
        iou: (..., N, M) CIoU（与 mask_pos 同形）
    返回: (..., N, M) bool
    """
    if mask_pos.shape[-1] > n_max_boxes:
        # 每 anchor 最多只需考虑 CIoU 最高的 n_max 个 GT，其余不可能胜出
        topk_iou, _ = iou.topk(n_max_boxes, dim=-1)
        mask_pos = mask_pos & (iou >= topk_iou[..., -1:])
    # 每 anchor 在其竞争 GT 中只保留 argmax
    max_iou, max_idx = iou.max(dim=-1)  # (..., N)
    one_max = F.one_hot(max_idx, iou.shape[-1]).bool()  # (..., N, M)
    return mask_pos & one_max & (iou == max_iou.unsqueeze(-1))


class TaskAlignedAssigner:
    """TAL + STAL（o2m: topk；o2o: topk_o2o + 二次 topk2 一对一）

    批量入口 forward_batch（训练用）；单图 forward 为兼容包装（同一实现路径）。
    """

    # 每分块 (B_chunk × N × M) 的元素预算：峰值 ≈ 若干同行 FP32 中间张量 × 4B
    CHUNK_ELEMS = 8_000_000

    def __init__(self, nc=80, topk=10, topk_o2o=7, topk2=1, alpha=0.5, beta=6.0, s_min=8.0, s_ref=16.0):
        self.nc = nc
        self.topk = topk
        self.topk_o2o = topk_o2o
        self.topk2 = topk2
        self.alpha = alpha
        self.beta = beta
        self.s_min = s_min
        self.s_ref = s_ref

    @torch.no_grad()
    def forward_batch(self, pred_boxes, pred_scores, anchors, strides, gt, gt_mask, one2one=False):
        """批量分配（B 维一次算完，分块控显存）

        Args:
            pred_boxes: (B, 4, N) ltrb 距离（grid 单位）
            pred_scores: (B, nc, N) 分类 logits（调用方负责 detach，官方口径）
            anchors: (2, N) anchor 中心 / strides: (1, N) 逐点 stride（grid 单位）
            gt: (B, M, 5) [cls, x1, y1, x2, y2] 像素坐标（短图右侧 padding）
            gt_mask: (B, M) bool，True = 有效 GT
            one2one: True 走 o2o 二次 top-k（真一对一）

        Returns:
            dict: fg_mask (B, N) / target_scores (B, N, nc) / target_ltrb (B, N, 4) /
                  target_boxes (B, N, 4) / n_pos (B,) long
        """
        B, _, N = pred_boxes.shape
        nc = pred_scores.shape[1]
        device = pred_boxes.device

        out = {
            "fg_mask": torch.zeros(B, N, dtype=torch.bool, device=device),
            "target_scores": torch.zeros(B, N, nc, device=device),
            "target_ltrb": torch.zeros(B, N, 4, device=device),
            "target_boxes": torch.zeros(B, N, 4, device=device),
            "n_pos": torch.zeros(B, dtype=torch.long, device=device),
        }
        if gt.shape[1] == 0 or not gt_mask.any():
            return out

        M = gt.shape[1]
        counts = gt_mask.sum(1).tolist()  # 唯一一次同步：每图 GT 数取回 CPU，供分块裁剪
        chunk = max(1, self.CHUNK_ELEMS // max(N * M, 1))
        for i in range(0, B, chunk):
            j = min(i + chunk, B)
            m = max(counts[i:j])  # 块内裁剪 padding 列（纯 Python，无 device 同步）
            if m == 0:
                continue
            self._assign_chunk(pred_boxes[i:j], pred_scores[i:j], anchors, strides,
                               gt[i:j, :m], gt_mask[i:j, :m], one2one, out, i, j)
        out["n_pos"] = out["fg_mask"].sum(-1)
        return out

    def _assign_chunk(self, pb, ps, anchors, strides, gt, gt_mask, one2one, out, i, j):
        """分块核心：所有 (b, n, m) 几何量按掩码批量计算，结果写入 out 的 [i:j] 切片"""
        B, _, N = pb.shape
        M = gt.shape[1]
        stride = strides[0]  # (N,)
        gt_cls = gt[..., 0].long()  # (B, M)
        # 逐 anchor 网格单位下的 GT（每 anchor 用自身 stride 归一）
        gt_grid = gt[:, None, :, 1:] / stride[None, :, None, None]  # (B, N, M, 4)

        # 预测框解码（逐 anchor 网格单位）: xyxy = [ax-l, ay-t, ay+r, ay+b]
        ax, ay = anchors[0], anchors[1]  # (N,)
        pred_xyxy = torch.stack(
            [ax - pb[:, 0], ay - pb[:, 1], ax + pb[:, 2], ay + pb[:, 3]], 2
        )  # (B, N, 4)

        # STAL 候选筛选：短边 < s_min 的 GT 按中心扩到 s_ref（仅此测试用代理框）
        w, h = gt_grid[..., 2] - gt_grid[..., 0], gt_grid[..., 3] - gt_grid[..., 1]  # (B, N, M)
        s_min_g = (self.s_min / stride)[None, :, None]
        s_ref_g = (self.s_ref / stride)[None, :, None]
        widen_w, widen_h = w < s_min_g, h < s_min_g
        gx1 = torch.where(widen_w, (gt_grid[..., 0] + gt_grid[..., 2] - s_ref_g) / 2, gt_grid[..., 0])
        gy1 = torch.where(widen_h, (gt_grid[..., 1] + gt_grid[..., 3] - s_ref_g) / 2, gt_grid[..., 1])
        gx2 = torch.where(widen_w, (gt_grid[..., 0] + gt_grid[..., 2] + s_ref_g) / 2, gt_grid[..., 2])
        gy2 = torch.where(widen_h, (gt_grid[..., 1] + gt_grid[..., 3] + s_ref_g) / 2, gt_grid[..., 3])
        candidate = (ax[None, :, None] >= gx1) & (ax[None, :, None] <= gx2) & \
                    (ay[None, :, None] >= gy1) & (ay[None, :, None] <= gy2)  # (B, N, M)
        candidate = candidate & gt_mask[:, None, :]  # padding 列不参与

        # 对齐度：sigmoid(cls_j score)^alpha * CIoU^beta（原始框）
        ciou = bbox_iou_torch(pred_xyxy[:, :, None, :], gt_grid)  # (B, N, M)
        score_gt_cls = ps.permute(0, 2, 1).gather(2, gt_cls[:, None, :].expand(B, N, M))  # (B, N, M)
        # CIoU 可为负，浮点指数幂对负底数产生 NaN -> 先 clamp；sigmoid 项 clamp 为
        # 反向 s^(alpha-1) 在 s 下溢到 0 时防 inf（与上面 CIoU 的 0/0 修复互为双保险）
        align = score_gt_cls.sigmoid().clamp_min(1e-8).pow(self.alpha) * ciou.clamp_min(0).pow(self.beta)
        align = align * candidate

        # per-GT top-k 正样本。固定 k 与旧逐图版 min(k, 最大候选数) 等价：非候选 align
        # 恒为 0，候选不足 k 时第 k 大值落回 0，阈值 ≥0 再 & candidate 即"全部候选"
        k = min(self.topk_o2o if one2one else self.topk, N)
        topk_align, _ = align.topk(k, dim=1)  # (B, k, M)（N 维 top-k）
        mask_pos = (align >= topk_align[:, -1:, :]) & candidate

        if one2one:
            # 二次 top-k：per-GT 按 CIoU 只留 topk2（真一对一）
            ciou_masked = ciou * mask_pos
            top2, _ = ciou_masked.topk(min(self.topk2, N), dim=1)
            mask_pos = mask_pos & (ciou_masked >= top2[:, -1:, :])

        # 多 GT 竞争消解
        mask_pos = select_highest_overlaps(mask_pos, ciou)

        fg_mask = mask_pos.any(dim=2)  # (B, N)
        if not fg_mask.any():
            return
        gt_idx = mask_pos.float().argmax(dim=2)  # (B, N)（非 fg 行取 0，下方被 fg 掩码清零）

        # 软标签：per-GT 归一化 align
        align_sum = align.masked_fill(~mask_pos, 0).sum(1).clamp_min(1e-16)  # (B, M)
        soft = align / align_sum[:, None, :]  # (B, N, M)
        soft = soft * mask_pos

        cls_of = gt_cls.gather(1, gt_idx)  # (B, N)
        val = soft.gather(2, gt_idx[:, :, None]).squeeze(2) * fg_mask  # (B, N)
        out["target_scores"][i:j].scatter_(2, cls_of[:, :, None], val[:, :, None])

        # 回归目标（逐 anchor grid 单位，原始框）：l = ax-gx1, t = ay-gy1, r = gx2-ax, b = gy2-ay
        g = gt_grid.gather(2, gt_idx[:, :, None, None].expand(B, N, 1, 4)).squeeze(2)  # (B, N, 4)
        m4 = fg_mask[:, :, None]
        out["target_ltrb"][i:j] = torch.stack(
            [ax[None, :] - g[..., 0], ay[None, :] - g[..., 1], g[..., 2] - ax[None, :], g[..., 3] - ay[None, :]], -1
        ) * m4
        out["target_boxes"][i:j] = g * m4
        out["fg_mask"][i:j] = fg_mask

    def forward(self, pred_boxes, pred_scores, anchors, strides, gt, one2one=False):
        """单图分配（兼容入口，与 forward_batch 同一实现路径）

        Args:
            pred_boxes: (4, N) ltrb 距离 / pred_scores: (nc, N) / gt: (M, 5) 像素坐标
        """
        gt_mask = torch.ones(1, gt.shape[0], dtype=torch.bool, device=gt.device)
        out = self.forward_batch(pred_boxes[None], pred_scores[None], anchors, strides,
                                 gt[None], gt_mask, one2one=one2one)
        return {
            "fg_mask": out["fg_mask"][0],
            "target_scores": out["target_scores"][0],
            "target_ltrb": out["target_ltrb"][0],
            "target_boxes": out["target_boxes"][0],
            "n_pos": int(out["n_pos"][0]),
        }
