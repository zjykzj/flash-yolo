"""双分支训练损失 + ProgLoss 权重调度

每图每分支：
    L = box_gain*L_ciou + cls_gain*L_bce + dfl_gain*L_l1
        L_ciou: 正样本 (1-CIoU)，按软标签加权，除以正样本软分数总和（per-GT 归一化后即 GT 数）
        L_bce:  全 anchor BCEWithLogits 软目标（均值口径）
        L_l1:   正样本 ltrb 距离 L1（grid 单位，均值口径；dfl 槽位复用，reg_max=1 无 DFL）
总损失 = alpha*L_o2m + (1-alpha)*L_o2o（ProgLoss，alpha 逐 epoch 0.8 -> 0.1）；
o2o 分支 cls 额外乘 cls_w。损失按 batch 求和（x batch），配合 nbs/accum 口径
（梯度尺度与物理 batch 无关）。
"""

import torch
import torch.nn.functional as F

from train.assigner import TaskAlignedAssigner, bbox_iou_torch
from utils.anchors import make_anchors

__all__ = ["ComputeLoss"]


class ComputeLoss:
    """YOLO26 双头损失（o2m + o2o，ProgLoss 渐进权重）"""

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
            items 为 per-image 均值口径的分解字典（日志用）
        """
        anchors, strides = self._anchors(imgsz)
        total = torch.zeros((), device=self.device)
        items = {"box": 0.0, "cls": 0.0, "dfl": 0.0, "o2m": 0.0, "o2o": 0.0}
        has_o2o = "one2one" in preds

        for branch, name in (("one2many", "o2m"), ("one2one", "o2o")):
            if branch not in preds:
                continue
            weight = self.alpha if name == "o2m" else 1.0 - self.alpha
            if name == "o2m" and not has_o2o:
                weight = 1.0  # 无 o2o 分支时 o2m 独占
            boxes = preds[branch]["boxes"].float()  # (B, 4, N) ltrb grid 单位
            scores = preds[branch]["scores"].float()  # (B, nc, N)
            cls_w = self.cfg.cls_w if branch == "one2one" else 1.0
            l_box = torch.zeros((), device=self.device)
            l_cls = torch.zeros((), device=self.device)
            l_l1 = torch.zeros((), device=self.device)

            for bi in range(batch_size):
                gt_img = targets[targets[:, 0] == bi][:, 1:]
                a = self.assigner.forward(boxes[bi], scores[bi], anchors, strides, gt_img, one2one=(branch == "one2one"))
                # 全 anchor 软目标 BCE（无正样本时目标全零，同样有限）
                l_cls = l_cls + F.binary_cross_entropy_with_logits(scores[bi].T, a["target_scores"], reduction="mean") * cls_w
                if a["n_pos"] == 0:
                    continue
                fg = a["fg_mask"]
                pred_xyxy = torch.stack(
                    [anchors[0] - boxes[bi][0], anchors[1] - boxes[bi][1], anchors[0] + boxes[bi][2], anchors[1] + boxes[bi][3]], 1
                )  # (N, 4) 逐 anchor 解码
                w = a["target_scores"][fg].sum(-1)  # (n_pos,) 软分数权重（per-GT 归一化）
                ciou = bbox_iou_torch(pred_xyxy[fg], a["target_boxes"][fg], ciou=True)
                l_box = l_box + ((1.0 - ciou) * w).sum() / w.sum().clamp_min(1.0)
                l_l1 = l_l1 + (boxes[bi][:, fg].T - a["target_ltrb"][fg]).abs().sum(-1).mean()

            branch_loss = (self.cfg.box_gain * l_box + self.cfg.cls_gain * l_cls + self.cfg.dfl_gain * l_l1) / batch_size
            total = total + weight * branch_loss * batch_size
            items[name] = branch_loss.detach().item()
            items["box"] += l_box.detach().item() / batch_size
            items["cls"] += l_cls.detach().item() / batch_size
            items["dfl"] += l_l1.detach().item() / batch_size

        return total, items

    __call__ = forward  # 普通类不自动调用 forward，显式别名
