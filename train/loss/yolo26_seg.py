"""YOLO26-seg 损失：检测损失（复用 ComputeLoss）+ 掩码 BCE 分支 + semseg 辅助项

按 Apache-2.0 项目约束：以下为对官方 v8SegmentationLoss 行为的独立重写，未复制 AGPL 源码。
口径对齐官方 v8SegmentationLoss（ultralytics 8.4.173）：
- 检测三项（box/cls/l1）与 ProgLoss 双分支权重完全复用 ComputeLoss；
- 掩码项只对正样本计算：系数 @ 原型 -> BCE(logits, GT mask) -> 按责任 GT 框裁剪 ->
  逐实例像素均值 / 归一化框面积 -> 全 batch 求和 / 正样本总数；
- 掩码增益 = cfg.box_gain（官方 `loss[1] *= hyp.box`），再按本地 `×batch_size / batch_size`
  口径接入（与检测项一致）；
- 无正样本时加 `(proto*0).sum() + (coeff*0).sum()` 保持图连通（官方同款）。

o2m/o2o 两个分支各用自己的系数/原型（o2o 的 proto 在 head 里已 detach），ProgLoss 的
alpha 权重作用于整条分支损失（与官方 E2ELoss 包住 v8SegmentationLoss 一致）。
"""

import torch
import torch.nn.functional as F

from train.loss.yolo26_detect import ComputeLoss

__all__ = ["ComputeLossSeg"]

# 掩码损失分块预算：每块 gather 的 proto 约 4 * nm * mh * mw 字节
_MASK_LOSS_CHUNK_ELEMS = 32_000_000  # ~128 MiB float32


def _crop_mask_float(loss, boxes):
    """框外清零（loss (n,h,w) float，boxes (n,4) 为掩码网格坐标）"""
    _, h, w = loss.shape
    x1, y1, x2, y2 = boxes.unbind(1)
    cols = (torch.arange(w, device=loss.device)[None, :] >= x1[:, None]) & \
           (torch.arange(w, device=loss.device)[None, :] < x2[:, None])
    rows = (torch.arange(h, device=loss.device)[None, :] >= y1[:, None]) & \
           (torch.arange(h, device=loss.device)[None, :] < y2[:, None])
    return loss * cols[:, None, :].to(loss.dtype) * rows[:, :, None].to(loss.dtype)


class ComputeLossSeg(ComputeLoss):
    """YOLO26-seg 双头损失（检测 + 掩码 + 官方 semseg 辅助项）"""

    item_keys = ("box", "seg", "cls", "l1", "o2m", "o2o", "sem")

    def __init__(self, cfg, head, device):
        super().__init__(cfg, head, device)
        self.overlap = bool(getattr(cfg, "overlap_mask", True))

    def _extra_branch_terms(self, branch, preds_b, assignment, masks, targets, strides, batch_size, imgsz):
        if masks is None:
            raise ValueError("ComputeLossSeg requires GT masks (masks=None)")
        coeff = preds_b["mask_coefficient"].float().permute(0, 2, 1)  # (B, N, nm)
        proto = preds_b["proto"]
        if isinstance(proto, tuple):  # 官方风格：proto 与 semantic 一起返回
            proto, semantic = proto
        else:
            semantic = preds_b.get("semantic")
        proto = proto.float()                                         # (B, nm, h, w)
        if tuple(masks.shape[-2:]) != tuple(proto.shape[-2:]):
            proto = F.interpolate(proto, size=tuple(masks.shape[-2:]), mode="bilinear", align_corners=False)

        B, N, nm = coeff.shape
        fg = assignment["fg_mask"].reshape(-1)  # (B*N,)
        pos = fg.nonzero(as_tuple=False).squeeze(1)  # (P,) 所有图的正样本 anchor
        if pos.numel() == 0:
            zero = (proto * 0).sum() + (coeff * 0).sum()
            if semantic is not None:
                zero = zero + (semantic * 0).sum()
            return self.cfg.box_gain * zero, {"seg": 0.0}

        # 掩码契约：overlap -> (B,h,w) 索引图；非 overlap -> (N_total,h,w) 与 targets 行一一对应
        if self.overlap:
            if masks.shape[0] != B:
                raise ValueError(f"overlap masks must have batch dim {B}, got {masks.shape[0]}")
        elif masks.shape[0] != targets.shape[0]:
            raise ValueError(f"non-overlap masks rows ({masks.shape[0]}) must match targets ({targets.shape[0]})")

        mh, mw = masks.shape[-2:]
        coeff_flat = coeff.reshape(B * N, nm)
        boxes_px = assignment["target_boxes"] * strides[0][None, :, None]  # grid -> 像素 (B,N,4)
        boxes_flat = boxes_px.reshape(B * N, 4)
        gt_idx_flat = assignment["target_gt_idx"].reshape(-1)

        img_idx = torch.div(pos, N, rounding_mode="floor")  # (P,) 每个正样本属于哪张图
        coeff_pos = coeff_flat[pos]                          # (P, nm)
        boxes_pos = boxes_flat[pos]                          # (P, 4)
        gt_idx_pos = gt_idx_flat[pos]                        # (P,) 责任 GT 索引

        norm = boxes_pos / imgsz
        area = ((norm[:, 2] - norm[:, 0]) * (norm[:, 3] - norm[:, 1])).clamp_min(1e-9)
        scale = torch.tensor([mw, mh, mw, mh], device=coeff.device, dtype=boxes_pos.dtype)
        mxyxy = norm * scale

        if not self.overlap:
            # 非 overlap 的 masks 行与 targets 行同序：全局行号 = 图偏移 + 图内 GT 索引
            counts = torch.bincount(targets[:, 0].long(), minlength=B)
            offsets = torch.cat([counts.new_zeros(1), counts.cumsum(0)[:-1]])
            row_idx = offsets[img_idx] + gt_idx_pos
            if row_idx.numel() and int(row_idx.max()) >= masks.shape[0]:
                raise ValueError("mask/target index out of range — masks are not aligned with targets")

        chunk = max(1, _MASK_LOSS_CHUNK_ELEMS // max(nm * mh * mw, 1))
        total = coeff.new_zeros(())
        n_pos = pos.numel()
        for s in range(0, n_pos, chunk):
            e = min(s + chunk, n_pos)
            proto_c = proto[img_idx[s:e]]  # (c, nm, mh, mw)
            pred = torch.bmm(coeff_pos[s:e].unsqueeze(1), proto_c.reshape(e - s, nm, -1))
            pred = pred.squeeze(1).reshape(-1, mh, mw)
            if self.overlap:
                gt = (masks[img_idx[s:e]] == (gt_idx_pos[s:e] + 1)[:, None, None]).to(pred.dtype)
            else:
                gt = masks[row_idx[s:e]].to(pred.dtype)
            loss = F.binary_cross_entropy_with_logits(pred, gt, reduction="none")
            loss = _crop_mask_float(loss, mxyxy[s:e])
            total = total + (loss.mean(dim=(1, 2)) / area[s:e]).sum()

        l_seg = total / n_pos
        extra = self.cfg.box_gain * l_seg * batch_size
        items = {"seg": l_seg.detach(), "sem": 0.0}
        if semantic is not None:
            sem_loss = self._semantic_loss(semantic, masks, targets, batch_size)
            extra = extra + self.cfg.box_gain * sem_loss * batch_size
            items["sem"] = sem_loss.detach()
        return extra, items

    def _semantic_target(self, masks, targets, batch_size):
        """掩码网格上的 one-hot 语义目标（背景全 0）；同一 batch 的两个分支复用同一份"""
        ref = getattr(self, "_sem_ref", None)
        if ref is not None and ref[0] is masks and ref[1] is targets:
            return self._sem_target_cache
        mh, mw = masks.shape[-2:]
        classes = targets[:, 1].long()
        target_batch = targets[:, 0].long()
        device = masks.device
        sem_idx = torch.zeros((batch_size, mh, mw), dtype=torch.long, device=device)
        present = torch.zeros((batch_size, mh, mw), dtype=torch.bool, device=device)
        for i in range(batch_size):
            sel = target_batch == i
            cls_i = classes[sel]
            if self.overlap:
                if masks.shape[0] != batch_size:
                    raise ValueError(f"overlap masks must have batch dim {batch_size}, got {masks.shape[0]}")
                present[i] = masks[i] != 0
                if cls_i.numel():
                    lookup = torch.zeros(cls_i.numel() + 1, dtype=torch.long, device=device)
                    lookup[1:] = cls_i
                    sem_idx[i] = lookup[masks[i].long()]
            else:
                m_i = masks[sel]
                if m_i.numel() == 0:
                    continue
                areas = m_i.sum(dim=(1, 2))
                weights = torch.where(m_i.bool(), areas[:, None, None], areas.max() + 1)
                key = weights.min(0).indices
                pres = m_i.any(0)
                sem_idx[i] = cls_i[key] * pres
                present[i] = pres
        target = F.one_hot(sem_idx, self.nc).permute(0, 3, 1, 2).float()
        target = target * present[:, None].to(target.dtype)
        self._sem_ref = (masks, targets)
        self._sem_target_cache = target
        return target

    def _semantic_loss(self, semantic, masks, targets, batch_size):
        """官方 semseg 辅助项：BCE + Dice（target 为掩码网格类别图，按 nearest 对齐到 P3）"""
        target = self._semantic_target(masks, targets, batch_size)
        pred = semantic.float()
        if tuple(target.shape[-2:]) != tuple(pred.shape[-2:]):
            target = F.interpolate(target, size=tuple(pred.shape[-2:]), mode="nearest")
        bce = F.binary_cross_entropy_with_logits(pred, target)
        p = pred.sigmoid()
        inter = (p * target).sum(dim=(2, 3))
        union = p.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2.0 * inter + 1.0) / (union + 1.0)  # MultiChannelDiceLoss(smooth=1)
        dice_loss = (1.0 - dice).mean(dim=1).mean()
        return 0.5 * bce + 0.5 * dice_loss
