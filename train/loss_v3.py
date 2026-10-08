"""YOLOv3-tiny 训练损失：BCE(obj) + BCE(cls) + CIoU（anchor-based 匹配，批量口径）

匹配（darknet 语义）：每 GT × 每级取该级 3 槽中「形状 IoU」（中心重合的宽高交并比）最大者
为唯一正样本；非正样本槽位中「预测解码框与任一 GT 的 IoU ≥ ignore_thresh(0.7)」者 obj 置
忽略（不罚），其余为负样本（obj 目标 0）。cls 独热 BCE 与 CIoU 仅正样本。

归一化沿用仓库口径：逐图均值（obj 按非忽略槽位数、cls/box 按正样本数），总损失
= Σ_级 Σ_图 (box_gain·box + obj_gain·obj + cls_gain·cls)——与 trainer 的 /batch 及
nbs/accum 梯度口径衔接。接口与 ComputeLoss 同构（item_keys / __call__），无 ProgLoss
（无 set_alpha，trainer 以 hasattr 守卫）。批量实现，无逐图循环。

同槽位冲突（两个 GT 命中同一 cell/slot）：scatter 后者胜；padding GT 写入哨兵列不参与。
"""

import torch
import torch.nn.functional as F

from model.head_v3 import decode_level
from train.assigner import bbox_iou_torch, padded_gt

__all__ = ["ComputeLossV3"]

_IGNORE_THRESH = 0.7  # darknet ignore_thresh：与 GT 高度重叠的非责任预测不罚 obj


class ComputeLossV3:
    """YOLOv3-tiny 损失（anchor-based，每 GT 每级至多一个正样本）"""

    item_keys = ("box", "obj", "cls")

    def __init__(self, cfg, head, device):
        self.cfg = cfg
        self.device = device
        self.nc = head.nc
        self.no = head.nc + 5
        self.na = getattr(head, "na", 3)
        self.obj_gain = getattr(cfg, "obj_gain", 1.0)
        self.strides = torch.as_tensor(head.stride, device=device, dtype=torch.float32)   # (nl,)
        self.anchors = torch.as_tensor(head.anchors, device=device, dtype=torch.float32)  # (nl, na, 2) 像素

    def _level_targets(self, lvl, pb, h, w, gt, gt_mask):
        """单级匹配目标（批量）

        Args:
            pb: (B, NA, 4) 预测解码框（像素 xyxy；仅用于忽略区判定）
            h, w: 该级特征图高宽（NA = na*h*w）
            gt: (B, M, 5) [cls, x1, y1, x2, y2] 像素；gt_mask: (B, M)
        Returns:
            pos (B, NA) bool / ignore (B, NA) bool / t_box (B, NA, 4) / t_cls (B, NA, nc)
        """
        b, m = gt.shape[0], gt.shape[1]
        na = self.na
        stride = self.strides[lvl]
        anchors = self.anchors[lvl]  # (na, 2)
        n_cells = na * h * w
        sentinel = n_cells  # padding GT 写哨兵列（随后丢弃），不擦除真实正样本

        if m == 0 or not gt_mask.any():
            zeros = torch.zeros(b, n_cells, device=gt.device)
            return (zeros.bool(), zeros.bool(),
                    zeros.new_zeros(b, n_cells, 4), zeros.new_zeros(b, n_cells, self.nc))

        # 形状 IoU（中心重合）：(B, M, na) → 每 GT 该级最优槽位
        gw = (gt[..., 3] - gt[..., 1]).clamp_min(0)
        gh = (gt[..., 4] - gt[..., 2]).clamp_min(0)
        inter = torch.minimum(gw[..., None], anchors[None, None, :, 0]) * \
                torch.minimum(gh[..., None], anchors[None, None, :, 1])
        union = gw[..., None] * gh[..., None] + \
                (anchors[None, None, :, 0] * anchors[None, None, :, 1]) - inter
        slot = (inter / (union + 1e-7)).argmax(-1)  # (B, M)

        # 责任格子：GT 中心所在 cell（clamp 到网格内）
        gx = (((gt[..., 1] + gt[..., 3]) * 0.5) / stride).long().clamp(0, w - 1)
        gy = (((gt[..., 2] + gt[..., 4]) * 0.5) / stride).long().clamp(0, h - 1)
        flat = slot * (h * w) + gy * w + gx  # (B, M)
        idx = torch.where(gt_mask, flat, torch.full_like(flat, sentinel))

        pos_ext = torch.zeros(b, n_cells + 1, dtype=torch.bool, device=gt.device)
        pos_ext.scatter_(1, idx, gt_mask)
        pos = pos_ext[:, :n_cells]

        t_box_ext = torch.zeros(b, n_cells + 1, 4, device=gt.device)
        t_box_ext.scatter_(1, idx[..., None].expand(-1, -1, 4), gt[..., 1:5])
        t_box = t_box_ext[:, :n_cells]

        t_cls_ext = torch.zeros(b, n_cells + 1, self.nc, device=gt.device)
        t_cls_ext.scatter_(1, idx[..., None].expand(-1, -1, self.nc),
                           F.one_hot(gt[..., 0].long(), self.nc).to(gt.dtype))
        t_cls = t_cls_ext[:, :n_cells]

        # 忽略区（darknet 口径）：非正样本槽位中预测框与任一 GT 的 IoU ≥ ignore_thresh
        with torch.no_grad():
            iou = bbox_iou_torch(pb.unsqueeze(2), gt[..., 1:5].unsqueeze(1), ciou=False)  # (B, NA, M)
            iou = iou * gt_mask[:, None, :]
            ignore = (~pos) & (iou.amax(2) >= _IGNORE_THRESH)
        return pos, ignore, t_box, t_cls

    def forward(self, preds, targets, batch_size, imgsz):
        """总损失（fp32 内部计算；批量口径）

        Args:
            preds: 每级原始输出列表 [(B, na*(5+nc), H, W), ...]（head._forward_train 口径）
            targets: (N, 6) [batch_idx, cls, x1, y1, x2, y2] letterbox 像素坐标
            batch_size: 本 batch 图像数
        Returns:
            (total, items)：total = Σ_级 Σ_图 加权损失（与 trainer 的 /batch 口径衔接），
            items = {"box","obj","cls"} 逐图均值（跨级平均，日志用，float）
        """
        gt, gt_mask = padded_gt(targets, batch_size)
        total = torch.zeros((), device=self.device)
        acc = []
        for lvl, raw in enumerate(preds):
            raw = raw.float()
            b, _, h, w = raw.shape
            boxes, obj_logits, cls_logits = decode_level(
                raw, self.anchors[lvl], self.strides[lvl], self.no, self.na)
            pb = boxes.reshape(b, -1, 4)                 # (B, NA, 4) 像素 xyxy
            obj = obj_logits.reshape(b, -1)              # (B, NA) logits
            cls = cls_logits.reshape(b, -1, self.nc)     # (B, NA, nc) logits
            pos, ignore, t_box, t_cls = self._level_targets(lvl, pb.detach(), h, w, gt, gt_mask)

            valid = ~ignore
            l_obj = (F.binary_cross_entropy_with_logits(obj, pos.float(), reduction="none") * valid
                     ).sum(-1) / valid.sum(-1).clamp_min(1)
            n_pos = pos.sum(-1).clamp_min(1)
            l_cls = (F.binary_cross_entropy_with_logits(cls, t_cls, reduction="none").sum(-1) * pos
                     ).sum(-1) / n_pos
            ciou = bbox_iou_torch(pb, t_box, ciou=True)  # (B, NA)（非正样本被 pos 掩码）
            l_box = ((1.0 - ciou) * pos).sum(-1) / n_pos

            total = total + (self.cfg.box_gain * l_box + self.obj_gain * l_obj
                             + self.cfg.cls_gain * l_cls).sum()
            acc.append(torch.stack([l_box.mean(), l_obj.mean(), l_cls.mean()]))  # (3,)

        # 一次性同步取日志值（每 iter 仅此一处 device->host）
        vals = torch.stack(acc).mean(0).tolist() if acc else [0.0] * len(self.item_keys)
        return total, dict(zip(self.item_keys, vals))

    __call__ = forward  # 普通类不自动调用 forward，显式别名
