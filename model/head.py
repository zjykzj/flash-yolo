"""YOLO26 检测头：双分支（o2m + o2o）、无 DFL（reg_max=1）、E2E 两阶段 top-k 解码

属性树（与官方 state_dict 对齐）：
    cv2.{i}.{0,1,2}          o2m box 分支（i 为检测层）
    cv3.{i}.{0.0,0.1,1.0,1.1,2}  o2m cls 分支
    one2one_cv2 / one2one_cv3    o2o 分支（结构与上相同）
"""

import copy
import math

import torch
import torch.nn as nn

from model.basic import Conv, DWConv
from utils.anchors import make_anchors, dist2bbox

__all__ = ["Detect"]


class Detect(nn.Module):
    """YOLO26 Detect 头

    推理两种模式：
        end2end=True  -> 前向直接输出 (B, max_det, 6) = [x1, y1, x2, y2, conf, cls]（NMS-free）
        end2end=False -> 输出 (B, 4+nc, N) 原始预测（ltrb + logits），NMS 在外部处理
    训练模式（model.train() 时自动切换）：双分支原始输出 dict（见 _forward_train）
    """

    max_det = 300

    def __init__(self, nc=80, reg_max=1, end2end=True, ch=()):
        super().__init__()
        self.nc = nc
        self.nl = len(ch)
        self.reg_max = reg_max
        self.no = nc + reg_max * 4
        self.stride = torch.zeros(self.nl)
        c2 = max(16, ch[0] // 4, reg_max * 4)  # box 分支隐藏通道（n 档 = 16）
        c3 = max(ch[0], min(nc, 100))          # cls 分支隐藏通道（n 档 = 80）
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * reg_max, 1)) for x in ch
        )
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                nn.Conv2d(c3, nc, 1),
            )
            for x in ch
        )
        self.dfl = nn.Identity()  # reg_max=1 -> 无 DFL
        self._end2end = False
        if end2end:
            self.one2one_cv2 = copy.deepcopy(self.cv2)
            self.one2one_cv3 = copy.deepcopy(self.cv3)

    # ---- 推理模式选择 ----
    @property
    def end2end(self):
        """是否走 o2o NMS-free 路径"""
        return getattr(self, "one2one_cv2", None) is not None and (self._end2end or self.cv2 is None)

    @end2end.setter
    def end2end(self, value):
        self._end2end = value

    def bias_init(self, imgsz=640):
        """初始化末层偏置（需先设置 stride；从零训练时使用，加载官方权重后会被覆盖）

        cls 先验 = 每图 5 个目标摊到 nc 类与 (imgsz/stride)² 个格子上；imgsz 取训练输入尺寸
        （官方实现写死 640，非 640 从零训练时先验偏 (640/imgsz)² 倍，此处按实际尺寸算）。
        """
        heads = [(self.cv2, self.cv3)]
        if getattr(self, "one2one_cv2", None) is not None:
            heads.append((self.one2one_cv2, self.one2one_cv3))
        for box_head, cls_head in heads:
            for i in range(self.nl):
                box_head[i][2].bias.data[:] = 2.0
                cls_head[i][2].bias.data[: self.nc] = math.log(5 / self.nc / (imgsz / self.stride[i]) ** 2)

    # ---- 前向 ----
    def forward(self, x):
        if self.training and getattr(self, "one2one_cv2", None) is not None:
            return self._forward_train(x)
        if self.end2end:
            y = self._forward_head(x, self.one2one_cv2, self.one2one_cv3)
            return self._e2e_postprocess(y, x)
        y = self._forward_head(x, self.cv2, self.cv3)
        return torch.cat((y["boxes"], y["scores"]), 1)  # (B, 4+nc, N) 原始输出

    def _forward_train(self, x):
        """训练模式：双分支原始输出 dict {"one2many": …, "one2one": …}

        o2o 分支吃 detached 特征——梯度只经 o2m 分支流向 backbone/neck，
        o2o 头仅训练自身卷积权重（官方口径）。
        """
        o2m = self._forward_head(x, self.cv2, self.cv3)
        x_detached = [feat.detach() for feat in x]
        o2o = self._forward_head(x_detached, self.one2one_cv2, self.one2one_cv3)
        return {"one2many": o2m, "one2one": o2o}

    def _forward_head(self, x, box_head, cls_head):
        bs = x[0].shape[0]
        boxes = torch.cat([box_head[i](x[i]).reshape(bs, 4 * self.reg_max, -1) for i in range(self.nl)], dim=-1)
        scores = torch.cat([cls_head[i](x[i]).reshape(bs, self.nc, -1) for i in range(self.nl)], dim=-1)
        return {"boxes": boxes, "scores": scores}

    def _e2e_postprocess(self, y, feats):
        """E2E 解码：ltrb -> xyxy（像素坐标）-> 两阶段 top-k -> (B, max_det, 6)"""
        anchors, strides = make_anchors(feats, self.stride, 0.5)
        boxes = dist2bbox(y["boxes"], anchors.unsqueeze(0)) * strides  # (B, 4, N) xyxy
        boxes = boxes.permute(0, 2, 1)                                # (B, N, 4)
        scores = y["scores"].permute(0, 2, 1).sigmoid()               # (B, N, nc)

        n = scores.shape[1]
        k = min(self.max_det, n)
        # 第一轮：每 anchor 取最大类分，选 top-k anchor
        ori_idx = scores.max(dim=-1)[0].topk(k, dim=1)[1]            # (B, k)
        cand = scores.gather(1, ori_idx.unsqueeze(-1).expand(-1, -1, self.nc))
        # 第二轮：在 k*80 候选中取全局 top-k（同一 anchor 可带不同类别出现）
        top_scores, idx = cand.flatten(1).topk(k, dim=1)
        cls = (idx % self.nc).float().unsqueeze(-1)                  # (B, k, 1)
        anchor_idx = ori_idx.gather(1, idx // self.nc)               # (B, k)
        out_boxes = boxes.gather(1, anchor_idx.unsqueeze(-1).expand(-1, -1, 4))
        return torch.cat((out_boxes, top_scores.unsqueeze(-1), cls), dim=-1)  # (B, k, 6)
