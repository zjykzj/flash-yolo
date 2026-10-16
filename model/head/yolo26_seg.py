"""YOLO26 分割头：Detect + mask 系数分支（cv4）+ 原型生成（Proto26）

结构 = yolo26_detect.py 的 Detect 逐行同构，只多两个分割专属件：
    cv4.{i}.{0,1,2} / one2one_cv4.{i}.{0,1,2}   每级 mask 系数分支（nm=32）
    proto.*                                     原型生成 (B, nm, H/4, W/4)

推理输出 = 检测输出末尾追加 nm 个系数通道（E2E: (B, max_det, 6+nm)；raw: (B, 4+nc+nm, N)），
原型单独一路返回——调用方按 `(outputs, proto)` 取值。
E2E 的两阶段 top-k 复用 Detect._e2e_postprocess（唯一实现），系数按同一次选择取回。

尚未实现的部分（见开发计划）：mask 损失（BCE + 原型组合）与训练期 mask 指标；
本文件只负责结构与前向契约。
"""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.basic import Conv
from model.head.yolo26_detect import Detect

__all__ = ["Proto26", "Segment26"]


class Proto26(nn.Module):
    """mask 原型生成器：多尺度融合 + 经典 Proto 解码

    融合：P4/P5 经 1×1 投影到 P3 通道数后按 nearest 上采样（×2 / ×4）与 P3 相加
    （官方以固定 scale_factor 表达，保持动态 shape 导出友好）。
    解码：3×3 -> ConvTranspose2d(×2) -> 3×3 -> 1×1，输出 (B, nm, H/4, W/4)（640 输入 = 160²）。

    semseg：官方 ckpt 携带的训练期辅助语义头。**官方训练从未使用它**
    （官方 train_metrics["val/sem_loss"] 恒为 0），保留子模块只为 strict load 零映射；
    本实现在前向里不计算它——推理零成本。将来若实现语义辅助损失再补分支。
    """

    def __init__(self, ch=(), c_=256, c2=32, nc=80):
        super().__init__()
        self.cv1 = Conv(c_, c_, 3)
        self.upsample = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)
        self.cv2 = Conv(c_, c_, 3)
        self.cv3 = Conv(c_, c2)
        self.feat_refine = nn.ModuleList(Conv(x, ch[0], 1) for x in ch[1:])
        self.feat_fuse = Conv(ch[0], c_, 3)
        self.semseg = nn.Sequential(Conv(ch[0], c_, 3), Conv(c_, c_, 3), nn.Conv2d(c_, nc, 1))

    def forward(self, x):
        feat = x[0]
        for i, refine in enumerate(self.feat_refine):
            feat = feat + F.interpolate(refine(x[i + 1]), scale_factor=2 ** (i + 1), mode="nearest")
        return self.cv3(self.cv2(self.upsample(self.cv1(self.feat_fuse(feat)))))


class Segment26(Detect):
    """YOLO26 分割头（Segment26 = Detect + mask 系数 + 原型）

    输出契约：
        eval（end2end）: ((B, max_det, 6 + nm), proto)
        eval（raw）    : ((B, 4 + nc + nm, N), proto)
        train          : 每分支 dict 追加 mask_coefficient 与 proto（o2o 分支 proto 走 detach，
                         与特征 detach 同口径：梯度只经 o2m 流回原型）
    """

    def __init__(self, nc=80, nm=32, npr=256, reg_max=1, end2end=True, ch=()):
        super().__init__(nc, reg_max, end2end, ch)
        self.nm = nm
        self.npr = npr
        c4 = max(ch[0] // 4, self.nm)
        self.cv4 = nn.ModuleList(
            nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nm, 1)) for x in ch
        )
        self.proto = Proto26(ch, npr, nm, nc)
        if end2end:
            self.one2one_cv4 = copy.deepcopy(self.cv4)

    def forward(self, x):
        proto = self.proto(x)
        if self.training and getattr(self, "one2one_cv2", None) is not None:
            return self._forward_train(x, proto)
        if self.end2end:
            y = self._forward_head_seg(x, self.one2one_cv2, self.one2one_cv3, self.one2one_cv4)
            return self._e2e_postprocess(y, x), proto
        y = self._forward_head_seg(x, self.cv2, self.cv3, self.cv4)
        return torch.cat((y["boxes"], y["scores"], y["mask_coefficient"]), 1), proto

    def _forward_head_seg(self, x, box_head, cls_head, mask_head):
        """Detect._forward_head + 逐级 mask 系数（拼接成 (B, nm, N)）"""
        y = self._forward_head(x, box_head, cls_head)
        bs = x[0].shape[0]
        y["mask_coefficient"] = torch.cat(
            [mask_head[i](x[i]).reshape(bs, self.nm, -1) for i in range(self.nl)], dim=-1
        )
        return y

    def _forward_train(self, x, proto):
        """训练模式：双分支原始输出 + 原型（o2o 分支的 proto 走 detach——与特征 detach 同口径）

        bias_init（继承自 Detect）只覆盖 cv2/cv3——cv4 不设先验（官方口径）。
        """
        o2m = self._forward_head_seg(x, self.cv2, self.cv3, self.cv4)
        x_detached = [feat.detach() for feat in x]
        o2o = self._forward_head_seg(x_detached, self.one2one_cv2, self.one2one_cv3, self.one2one_cv4)
        o2m["proto"], o2o["proto"] = proto, proto.detach()
        return {"one2many": o2m, "one2one": o2o}
