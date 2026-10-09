"""双分支训练损失 + ProgLoss 权重调度（批量口径，归一化与官方 ultralytics 逐项对齐）

每图每分支（三项共用归一化分母 Σt = 全 batch 软标签之和，官方 `target_scores_sum`）：
    L = box_gain*L_ciou + cls_gain*L_bce + dfl_gain*L_l1
        L_ciou: Σ_fg (1-CIoU)·t / Σt（软标签加权；无正样本时为 0）
        L_bce:  Σ_all BCEWithLogits(软目标 t) / Σt
                （官方分母是 Σtarget_scores 而非元素数 B·N·nc——曾用全元素均值把
                cls 梯度稀释 ~10^6 倍，分类头整轮不动、mAP 恒 0，见 CHANGELOG）
        L_l1:   Σ_fg mean_4(|Δltrb|·stride/imgsz)·t / Σt
                （ltrb 按 stride/imgsz 归一化后再取 4 边均值；reg_max=1 无 DFL）
总损失 = alpha*L_o2m + (1-alpha)*L_o2o（ProgLoss，alpha 逐 epoch 0.8 -> 0.1）；两项同增益，
不额外加权（官方 E2ELoss 口径）。损失按 batch 求和（x batch），配合 nbs/accum 口径
（梯度尺度与物理 batch 无关）。

批量口径：所有项对全 batch 一次算完（(B, N[, nc]) 掩码张量）。分配输入 detach 且
assigner 内部 no_grad（官方口径）：标签分配是离散操作，梯度只经损失项回流。
tests/test_loss_parity.py 以固定输入对官方 E2ELoss 做逐项数值 parity。
"""

import torch
import torch.nn.functional as F

from train.assigner import TaskAlignedAssigner, bbox_iou_torch, padded_gt
from train.loss_v3 import ComputeLossV3
from utils.anchors import make_anchors

__all__ = ["ComputeLoss", "build_loss"]


class ComputeLoss:
    """YOLO26 双头损失（o2m + o2o，ProgLoss 渐进权重）"""

    item_keys = ("box", "cls", "l1", "o2m", "o2o")  # 训练日志/CSV 列（trainer 由此派生）

    def __init__(self, cfg, head, device):
        self.cfg = cfg
        self.device = device
        self.nc = head.nc
        self.strides = torch.as_tensor(head.stride, device=device, dtype=torch.float32)  # [8,16,32]
        self.alpha = cfg.prog_alpha_init
        self.assigner = TaskAlignedAssigner(
            nc=head.nc,
            topk=cfg.topk,
            topk_o2o=cfg.topk_o2o,
            topk2=cfg.topk2,
            alpha=cfg.tal_alpha,
            beta=cfg.tal_beta,
            s_min=cfg.stal_s_min,
            s_ref=cfg.stal_s_ref,
        )

    def set_alpha(self, epoch, epochs):
        """ProgLoss：alpha(t) = max(1 - t/max(E-1,1), 0)*(a_init - a_final) + a_final（每 epoch 一次）"""
        t = epoch
        self.alpha = max(1.0 - t / max(epochs - 1, 1), 0.0) * (self.cfg.prog_alpha_init - self.cfg.prog_alpha_final) + self.cfg.prog_alpha_final

    def _anchors(self, imgsz):
        """各层 anchor 与逐点 stride（由 stride + 输入尺寸推导，multi_scale 关闭时精确）"""
        feats = [torch.zeros(1, 1, imgsz // int(s), imgsz // int(s), device=self.device) for s in self.strides]
        return make_anchors(feats, self.strides, 0.5)

    def forward(self, preds, targets, batch_size, imgsz):
        """总损失（fp32 内部计算，兼容 AMP 下的 fp16 输入）

        Args:
            preds: head 训练输出 {"one2many": {"boxes","scores"}, "one2one": ...}
            targets: (N, 6) [batch_idx, cls, x1, y1, x2, y2] 像素坐标
            batch_size: 本 batch 图像数
        Returns:
            (loss_total, items) — total 为 weight*Σ_img L_img（x batch 与 nbs/accum 口径一致），
            items 为 per-image 均值口径的分解字典（日志用，float）
        """
        anchors, strides = self._anchors(imgsz)
        gt, gt_mask = padded_gt(targets, batch_size)
        total = torch.zeros((), device=self.device)
        items = {"box": 0.0, "cls": 0.0, "l1": 0.0, "o2m": 0.0, "o2o": 0.0}
        has_o2o = "one2one" in preds

        for branch, name in (("one2many", "o2m"), ("one2one", "o2o")):
            if branch not in preds:
                continue
            weight = self.alpha if name == "o2m" else 1.0 - self.alpha
            if name == "o2m" and not has_o2o:
                weight = 1.0  # 无 o2o 分支时 o2m 独占
            boxes = preds[branch]["boxes"].float()  # (B, 4, N) ltrb grid 单位
            scores = preds[branch]["scores"].float()  # (B, nc, N)

            # 标签分配（detach + 内部 no_grad，官方口径）
            a = self.assigner.forward_batch(
                boxes.detach(), scores.detach(), anchors, strides, gt, gt_mask, one2one=(branch == "one2one")
            )
            fg = a["fg_mask"]  # (B, N)
            t = a["target_scores"]  # (B, N, nc) 软标签（官方口径，逐 GT 归一）
            t_sum = t.sum().clamp_min(1.0)  # 官方三项损失共用的归一化分母 Σtarget_scores
            w = t.sum(-1)  # (B, N) 逐 anchor 软权重（非 fg 恒 0）

            # 分类：官方 `bce.sum() / target_scores_sum`（×batch 维持 Σ_img 口径）
            l_cls = F.binary_cross_entropy_with_logits(
                scores.permute(0, 2, 1), t, reduction="sum"
            ) / t_sum * batch_size

            # 框：软加权 (1-CIoU) 求和 / Σt（官方 box 项同式；无正样本时自然为 0）
            ax, ay = anchors[0], anchors[1]
            pred_xyxy = torch.stack(
                [ax - boxes[:, 0], ay - boxes[:, 1], ax + boxes[:, 2], ay + boxes[:, 3]], 2
            )  # (B, N, 4)
            ciou = bbox_iou_torch(pred_xyxy, a["target_boxes"], ciou=True)  # (B, N)
            l_box = ((1.0 - ciou) * w * fg).sum() / t_sum * batch_size

            # l1：ltrb 按 stride/imgsz 归一化后取 4 边均值、软加权 / Σt（官方 reg_max=1 分支同式）
            d = (boxes.transpose(1, 2) - a["target_ltrb"]) * strides[0][None, :, None] / imgsz
            l_l1 = (d.abs().mean(-1) * w * fg).sum() / t_sum * batch_size

            branch_loss = (self.cfg.box_gain * l_box + self.cfg.cls_gain * l_cls + self.cfg.dfl_gain * l_l1) / batch_size
            total = total + weight * branch_loss * batch_size
            items[name] = items[name] + branch_loss
            items["box"] = items["box"] + l_box / batch_size
            items["cls"] = items["cls"] + l_cls / batch_size
            items["l1"] = items["l1"] + l_l1 / batch_size

        # 一次性同步取日志值（每 iter 仅此一处 device->host）
        keys = list(items)
        vals = torch.stack([
            torch.as_tensor(items[k], device=self.device, dtype=torch.float32).detach().reshape(()) for k in keys
        ]).tolist()
        return total, dict(zip(keys, vals))

    __call__ = forward  # 普通类不自动调用 forward，显式别名


_LOSS_BY_ARCH = {"yolo26": ComputeLoss, "yolov3-tiny": ComputeLossV3, "flash-yolo": ComputeLoss}


def build_loss(arch, cfg, head, device):
    """按架构分派损失实现（trainer 的单一入口；接口同构：item_keys / __call__）"""
    cls = _LOSS_BY_ARCH.get(arch)
    if cls is None:
        raise ValueError(f"unknown arch {arch!r}; available: {sorted(_LOSS_BY_ARCH)}")
    return cls(cfg, head, device)
