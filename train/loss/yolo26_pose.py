"""YOLO26-pose 损失：检测三项（复用 ComputeLoss）+ 关键点位置/可见性 + RLE

按 Apache-2.0 项目约束：以下为对官方 `PoseLoss26` / `KeypointLoss` / `RLELoss` 行为的
独立重写（依据论文与官方 8.4.173 的公式/张量契约，未复制 AGPL 源码）。

口径（与官方 pose loss 对齐）：
- 检测三项/ProgLoss 双分支权重完全复用 ComputeLoss（o2o 吃 detach 特征）；
- 关键点项只对正样本：关键点解码到 grid 坐标（anchor + t）后与 GT（像素 / stride）比较；
  KeypointLoss 用 COCO OKS sigmas、按责任 GT 框面积归一化，`(1-exp(-e))*kpt_mask` 的均值；
- 可见性项 = BCEWithLogits(pred_v, kpt_mask)（D=3 才有；D=2 无 v 通道）；
- RLE 项 = RealNVP 对归一化误差的 log_prob + sigma 的负对数似然（越界/NaN 过滤、
  error clamp ±100、loss clamp(min=0)，官方 `calculate_rle_loss` 同口径）；
- 增益：pose / kobj / rle（官方 pose ckpt 记录 24 / 4 / 1）；总损失按 `×batch_size`
  接入 ComputeLoss 的 Σ_img 口径（与检测/分割子类一致），items 为 per-image 均值。

无正样本时加 `(pred_kpts*0).sum()`（RLE 再加 flow 参数的零项）保持图连通，避免 DDP
unused-gradient 报错（官方同款）。
"""

import torch
import torch.nn as nn

from train.loss.yolo26_detect import ComputeLoss

__all__ = ["ComputeLossPose", "KeypointLoss", "RLELoss", "OKS_SIGMA", "RLE_WEIGHT"]

# COCO 17 关键点 OKS sigmas（官方 metrics.OKS_SIGMA / 10）
OKS_SIGMA = torch.tensor(
    [0.26, 0.25, 0.25, 0.35, 0.35, 0.79, 0.79, 0.72, 0.72, 0.62, 0.62, 1.07,
     1.07, 0.87, 0.87, 0.89, 0.89], dtype=torch.float32) / 10.0
# RLE 的逐关键点权重（官方 RLE_WEIGHT；仅 COCO 17 点时使用）
RLE_WEIGHT = torch.tensor(
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.2, 1.2, 1.5, 1.5, 1.0, 1.0, 1.2, 1.2, 1.5, 1.5],
    dtype=torch.float32)


class KeypointLoss(nn.Module):
    """关键点位置损失（官方 KeypointLoss 公式：OKS 尺度的 exp(-e) 形式）"""

    def __init__(self, sigmas):
        super().__init__()
        self.register_buffer("sigmas", torch.as_tensor(sigmas, dtype=torch.float32))

    def forward(self, pred_kpts, gt_kpts, kpt_mask, area):
        d = (pred_kpts[..., 0] - gt_kpts[..., 0]).pow(2) + \
            (pred_kpts[..., 1] - gt_kpts[..., 1]).pow(2)
        kpt_loss_factor = kpt_mask.shape[1] / (kpt_mask.sum(dim=1) + 1e-9)
        e = d / ((2 * self.sigmas).pow(2) * (area + 1e-9) * 2)  # 官方 cocoeval 口径
        return (kpt_loss_factor.view(-1, 1) * ((1 - torch.exp(-e)) * kpt_mask)).mean()


class RLELoss(nn.Module):
    """Residual Log-Likelihood Estimation loss（官方 RLELoss 公式）"""

    def __init__(self, use_target_weight=True):
        super().__init__()
        self.use_target_weight = use_target_weight

    def forward(self, sigma, log_phi, error, target_weight=None):
        loss = torch.log(sigma) - log_phi.unsqueeze(1)
        loss = loss + torch.log(sigma * 2) + torch.abs(error)
        if self.use_target_weight:
            if target_weight.dim() == 1:
                target_weight = target_weight.unsqueeze(1)
            loss = loss * target_weight
        loss = loss / len(loss)
        return loss.sum()


class ComputeLossPose(ComputeLoss):
    """YOLO26-pose 双头损失（检测 + 关键点位置/可见性 + RLE）"""

    item_keys = ("box", "pose", "kobj", "rle", "cls", "l1", "o2m", "o2o")

    def __init__(self, cfg, head, device):
        super().__init__(cfg, head, device)
        self.kpt_shape = tuple(head.kpt_shape)
        k = self.kpt_shape[0]
        is_coco = self.kpt_shape == (17, 3)
        sigmas = (OKS_SIGMA if is_coco else torch.ones(k) / k).to(device)
        self.keypoint_loss = KeypointLoss(sigmas).to(device)
        self.bce_pose = nn.BCEWithLogitsLoss()
        self.rle_loss = RLELoss(use_target_weight=True)
        self.rle_weight = (RLE_WEIGHT if is_coco else torch.ones(k)).to(device)
        self.flow_model = getattr(head, "flow_model", None)

    @staticmethod
    def _select_target_keypoints(keypoints, batch_idx, target_gt_idx, masks):
        """按 (图, GT 索引) 为每个 anchor 选责任 GT 关键点 -> (B, N, K, D)"""
        batch_idx = batch_idx.flatten().long()
        batch_size = masks.shape[0]
        n, k, d = keypoints.shape
        if n == 0:
            return keypoints.new_zeros((batch_size, target_gt_idx.shape[1], k, d))
        counts = torch.bincount(batch_idx, minlength=batch_size)
        max_k = int(counts.max())
        batched = keypoints.new_zeros((batch_size, max_k, k, d))
        offsets = torch.zeros(batch_size + 1, dtype=torch.long, device=keypoints.device)
        offsets.scatter_add_(0, batch_idx + 1, torch.ones_like(batch_idx))
        offsets = offsets.cumsum(0)
        within = torch.arange(len(batch_idx), device=keypoints.device) - offsets[batch_idx]
        batched[batch_idx, within] = keypoints
        idx = target_gt_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, k, d)
        return batched.gather(1, idx)

    def _calculate_rle_loss(self, pred_kpts, gt_kpts, kpt_mask):
        """官方 PoseLoss26.calculate_rle_loss 的等价实现（pred 末两维 = sigma）"""
        if self.flow_model is None:
            return pred_kpts[..., :0].sum()
        if not kpt_mask.any():
            return (pred_kpts[..., :0].sum()
                    + sum(p[..., :0].sum() for p in self.flow_model.parameters()))
        pred_vis = pred_kpts[kpt_mask]              # (M, 5)
        gt_vis = gt_kpts[kpt_mask]                  # (M, 3)
        pred_coords = pred_vis[:, :2]
        pred_sigma = pred_vis[:, -2:].sigmoid()
        gt_coords = gt_vis[:, :2]
        weight = self.rle_weight.unsqueeze(0).repeat(kpt_mask.shape[0], 1)[kpt_mask]
        error = (pred_coords - gt_coords) / (pred_sigma + 1e-9)
        valid = ~(torch.isnan(error) | torch.isinf(error)).any(dim=-1)
        if not valid.any():
            return pred_kpts[..., :0].sum()
        error = error[valid].clamp(-100, 100)
        pred_sigma = pred_sigma[valid]
        weight = weight[valid]
        log_phi = self.flow_model.log_prob(error)
        return self.rle_loss(pred_sigma, log_phi, error, weight).clamp(min=0)

    def _extra_branch_terms(self, branch, preds_b, assignment, masks, targets, strides, batch_size, imgsz,
                            keypoints=None):
        if keypoints is None:
            raise ValueError("ComputeLossPose requires GT keypoints (keypoints=None)")
        k, d = self.kpt_shape
        nk = k * d
        raw = preds_b["kpts"].float().permute(0, 2, 1)  # (B, N, nk)
        b, n, _ = raw.shape
        kpts = raw.reshape(b, n, k, d)
        raw_sigma = preds_b.get("kpts_sigma")
        sigma = (raw_sigma.float().permute(0, 2, 1).reshape(b, n, k, 2)
                 if raw_sigma is not None else None)
        anchors, strides = self._anchors(imgsz)
        pred_kpts = kpts.clone()
        pred_kpts[..., 0] = pred_kpts[..., 0] + anchors[0].view(1, -1, 1)
        pred_kpts[..., 1] = pred_kpts[..., 1] + anchors[1].view(1, -1, 1)

        fg = assignment["fg_mask"]  # (B, N)
        if not fg.any():
            zero = (pred_kpts * 0).sum()
            if sigma is not None:
                zero = zero + (sigma * 0).sum()
            gain = self.cfg.pose_gain + self.cfg.kobj_gain
            if sigma is not None:
                gain += self.cfg.rle_gain
            return zero * gain * batch_size, {"pose": 0.0, "kobj": 0.0, "rle": 0.0}

        selected = self._select_target_keypoints(keypoints, targets[:, 0],
                                                 assignment["target_gt_idx"], fg)
        gt_kpt = selected[fg].clone()  # (P, K, D) 原图像素
        pred_kpt = pred_kpts[fg]       # (P, K, D) grid 坐标
        stride_fg = strides.expand(b, -1)[fg]  # (P,)
        gt_kpt[..., :2] = gt_kpt[..., :2] / stride_fg[:, None, None]
        target_boxes = assignment["target_boxes"][fg]  # (P, 4) grid 单位
        area = ((target_boxes[:, 2] - target_boxes[:, 0])
                * (target_boxes[:, 3] - target_boxes[:, 1])).clamp_min(0).unsqueeze(-1)
        kpt_mask = (gt_kpt[..., 2] != 0) if d >= 3 else torch.ones(
            gt_kpt.shape[:2], dtype=torch.bool, device=gt_kpt.device)

        l_pose = self.keypoint_loss(pred_kpt, gt_kpt, kpt_mask, area)
        l_kobj = (self.bce_pose(pred_kpt[..., 2], kpt_mask.float())
                  if d >= 3 else pred_kpt.new_zeros(()))
        l_rle = pred_kpt.new_zeros(())
        if sigma is not None and self.flow_model is not None:
            l_rle = self._calculate_rle_loss(torch.cat([pred_kpt, sigma[fg]], dim=-1),
                                             gt_kpt, kpt_mask)

        extra = (self.cfg.pose_gain * l_pose + self.cfg.kobj_gain * l_kobj
                 + self.cfg.rle_gain * l_rle) * batch_size
        return extra, {"pose": l_pose.detach(), "kobj": l_kobj.detach(), "rle": l_rle.detach()}
