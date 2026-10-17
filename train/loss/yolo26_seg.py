"""YOLO26-seg 损失：检测损失（复用 ComputeLoss）+ 掩码 BCE 分支

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
    """YOLO26-seg 双头损失（检测 + 掩码）"""

    item_keys = ("box", "seg", "cls", "l1", "o2m", "o2o")

    def __init__(self, cfg, head, device):
        super().__init__(cfg, head, device)
        self.overlap = bool(getattr(cfg, "overlap_mask", True))

    def _extra_branch_terms(self, branch, preds_b, assignment, masks, targets, strides, batch_size, imgsz):
        if masks is None:
            raise ValueError("ComputeLossSeg requires GT masks (masks=None)")
        coeff = preds_b["mask_coefficient"].float().permute(0, 2, 1)  # (B, N, nm)
        proto = preds_b["proto"].float()                               # (B, nm, h, w)
        if tuple(masks.shape[-2:]) != tuple(proto.shape[-2:]):
            proto = F.interpolate(proto, size=tuple(masks.shape[-2:]), mode="bilinear", align_corners=False)

        fg = assignment["fg_mask"]  # (B, N)
        if not fg.any():
            zero = (proto * 0).sum() + (coeff * 0).sum()
            return self.cfg.box_gain * zero, {"seg": 0.0}

        mh, mw = masks.shape[-2:]
        boxes_px = assignment["target_boxes"] * strides[0][None, :, None]  # grid -> 像素 (B,N,4)
        gt_idx_all = assignment["target_gt_idx"]                           # (B,N) 每图 GT 索引
        target_batch = targets[:, 0].long()
        scale = torch.tensor([mw, mh, mw, mh], device=coeff.device, dtype=boxes_px.dtype)

        total = coeff.new_zeros(())
        n_pos = coeff.new_zeros(())
        for i in range(coeff.shape[0]):
            fg_i = fg[i]
            if not fg_i.any():
                continue
            idx = gt_idx_all[i][fg_i]                    # (n,) responsibility GT index
            boxes_i = boxes_px[i][fg_i]                  # (n,4) letterbox pixels
            norm = boxes_i / imgsz
            area = ((norm[:, 2] - norm[:, 0]) * (norm[:, 3] - norm[:, 1])).clamp_min(1e-9)
            mxyxy = norm * scale

            pred = coeff[i][fg_i] @ proto[i].reshape(proto.shape[1], -1)  # (n, mh*mw)
            pred = pred.reshape(-1, mh, mw)
            if self.overlap:
                gt = (masks[i] == (idx + 1)[:, None, None]).to(pred.dtype)
            else:
                sel = target_batch == i
                gt = masks[sel][idx].to(pred.dtype)

            loss = F.binary_cross_entropy_with_logits(pred, gt, reduction="none")
            loss = _crop_mask_float(loss, mxyxy)
            total = total + (loss.mean(dim=(1, 2)) / area).sum()
            n_pos = n_pos + fg_i.sum()

        l_seg = total / n_pos.clamp_min(1)
        return self.cfg.box_gain * l_seg * batch_size, {"seg": l_seg.detach()}
