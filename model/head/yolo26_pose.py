"""YOLO26 姿态估计头：Pose26 = Detect + 关键点分支 + RealNVP（RLE）

属性树（与官方 state_dict 对齐）：
    cv2.{i}.{0,1,2} / cv3.{i}.{...}       继承自 Detect 的框/类分支
    cv4.{i}.{0,1}                         每级关键点特征（两级 Conv，无 1×1 输出）
    cv4_kpts.{i}                          每级 1×1 -> nk（x,y,v 交错，nk = K*D）
    cv4_sigma.{i}                         每级 1×1 -> 2K（训练期 RLE 的 sigma，推理不用）
    one2one_cv*                            o2o 分支深拷贝（end2end）
    flow_model.*                          RealNVP 归一化流（训练期 RLE；推理不用，但权重里有）

推理输出契约：
    E2E（end2end=True）  : (B, max_det, 6 + nk) = [x1,y1,x2,y2,score,cls, kpts...]
                           kpts 与框共享同一次两阶段 top-k（e2e_topk 的 extra）。
    raw（end2end=False） : (B, 4 + nc + nk, N) = 已解码像素 xyxy + sigmoid 分数 + 解码关键点。
                           与官方 pose `_inference` 同口径（注意：与检测头的"原始 ltrb+logits"
                           契约不同——关键点解码需要 anchors/strides，留在图内更简单，也让
                           `--raw` 导出与官方 pose 输出逐位可比）。外部 NMS 只做框抑制。
    训练（model.train()）: 每分支 dict，含 boxes/scores/kpts（+ kpts_sigma）；o2o 吃 detach 特征。

关键点解码（官方 Pose26 口径）：`(t + anchor) * stride`，v 走 sigmoid——没有 v8 Pose 的
`2*t - 0.5` 口径。官方 release 代码在 export 分支用 view 表达，本实现同款两路。
"""

import copy
import math

import torch
import torch.nn as nn

from model.basic import Conv
from model.head.yolo26_detect import Detect, e2e_topk
from utils.anchors import dist2bbox, make_anchors

__all__ = ["RealNVP", "Pose26"]


class RealNVP(nn.Module):
    """RLE 用的 RealNVP 归一化流（6 层交替掩码 coupling）

    结构/键名与官方 `RealNVP` 一致：buffers loc/cov/mask + s/t 各 6 个 MLP
    （s 末尾多一个 Tanh）。推理前向不调用；训练期 RLELoss 用 `log_prob`。
    参考：RealNVP (arXiv:1605.08803) / RLE (arXiv:2107.11291)。
    """

    @staticmethod
    def nets():
        """scale 网络：2 -> 64 -> 64 -> 2，末层 Tanh"""
        return nn.Sequential(nn.Linear(2, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(),
                             nn.Linear(64, 2), nn.Tanh())

    @staticmethod
    def nett():
        """translation 网络：2 -> 64 -> 64 -> 2（无 Tanh）"""
        return nn.Sequential(nn.Linear(2, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(),
                             nn.Linear(64, 2))

    def __init__(self):
        super().__init__()
        # loc/cov 是旧版 checkpoint 的兼容 buffer（当前先验是闭式标准正态，不读它们）；
        # 保留注册才能 strict load 官方权重。
        self.register_buffer("loc", torch.zeros(2))
        self.register_buffer("cov", torch.eye(2))
        self.register_buffer("mask", torch.tensor([[0, 1], [1, 0]] * 3, dtype=torch.float32))
        self.s = nn.ModuleList([self.nets() for _ in range(len(self.mask))])
        self.t = nn.ModuleList([self.nett() for _ in range(len(self.mask))])
        self.init_weights()

    def init_weights(self):
        """官方初始化：所有 Linear 用 xavier_uniform_(gain=0.01)"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)

    def backward_p(self, x):
        """数据空间 -> 隐空间，并累加 log|det J|（x: (N, 2)）"""
        log_det_jacob, z = x.new_zeros(x.shape[0]), x
        for i in reversed(range(len(self.t))):
            z_ = self.mask[i] * z
            s = self.s[i](z_) * (1 - self.mask[i])
            t = self.t[i](z_) * (1 - self.mask[i])
            z = (1 - self.mask[i]) * (z - t) * torch.exp(-s) + z_
            log_det_jacob -= s.sum(dim=1)
        return z, log_det_jacob

    def log_prob(self, x):
        """标准正态先验下的 log p(x)（2 维闭式；fp32 防止 AMP 下 z² 溢出）"""
        if x.dtype == torch.float32 and self.s[0][0].weight.dtype != torch.float32:
            self.float()
        z, log_det = self.backward_p(x)
        return -0.5 * (z.float() ** 2).sum(-1) - math.log(2 * math.pi) + log_det


class Pose26(Detect):
    """YOLO26 姿态估计头（Pose26 = Detect + 关键点分支 + RealNVP）

    kpt_shape = (K, D)：COCO 为 (17, 3)，D=2 时只有 x,y、没有可见性通道。
    """

    task = "pose"
    export = False  # ONNX/TRT 导出时由 scripts/export.py 置 True（kpts_decode 走 view 分支）

    def __init__(self, nc=1, kpt_shape=(17, 3), reg_max=1, end2end=True, ch=()):
        super().__init__(nc, reg_max, end2end, ch)
        self.kpt_shape = tuple(int(v) for v in kpt_shape)
        if len(self.kpt_shape) != 2 or self.kpt_shape[0] <= 0 or self.kpt_shape[1] not in (2, 3):
            raise ValueError(f"kpt_shape must be (K, 2) or (K, 3), got {self.kpt_shape!r}")
        self.nk = self.kpt_shape[0] * self.kpt_shape[1]
        self.nk_sigma = self.kpt_shape[0] * 2
        # 官方 Pose26：c4 = max(ch0/4, K*(D+2))；D=3 时 5K，n 档 = 85
        c4 = max(ch[0] // 4, self.kpt_shape[0] * (self.kpt_shape[1] + 2))
        self.cv4 = nn.ModuleList(
            nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3)) for x in ch
        )
        self.cv4_kpts = nn.ModuleList(nn.Conv2d(c4, self.nk, 1) for _ in ch)
        self.cv4_sigma = nn.ModuleList(nn.Conv2d(c4, self.nk_sigma, 1) for _ in ch)
        self.flow_model = RealNVP()
        if end2end:
            self.one2one_cv4 = copy.deepcopy(self.cv4)
            self.one2one_cv4_kpts = copy.deepcopy(self.cv4_kpts)
            self.one2one_cv4_sigma = copy.deepcopy(self.cv4_sigma)

    # ---- 前向 ----
    def forward(self, x):
        if self.training and getattr(self, "one2one_cv2", None) is not None:
            return self._forward_train(x)
        if self.end2end:
            y = self._forward_head_pose(x, self.one2one_cv2, self.one2one_cv3,
                                        self.one2one_cv4, self.one2one_cv4_kpts, None)
            return self._e2e_postprocess_pose(y, x)
        y = self._forward_head_pose(x, self.cv2, self.cv3, self.cv4, self.cv4_kpts, None)
        return self._decode_pose(y, x)

    def _forward_head_pose(self, x, box_head, cls_head, pose_head, kpts_head, sigma_head):
        """Detect 的 boxes/scores + 关键点（训练时另加 sigma）"""
        y = self._forward_head(x, box_head, cls_head)
        bs = x[0].shape[0]
        feats = [pose_head[i](x[i]) for i in range(self.nl)]
        y["kpts"] = torch.cat(
            [kpts_head[i](feats[i]).reshape(bs, self.nk, -1) for i in range(self.nl)], dim=2
        )
        if self.training and sigma_head is not None:
            y["kpts_sigma"] = torch.cat(
                [sigma_head[i](feats[i]).reshape(bs, self.nk_sigma, -1) for i in range(self.nl)],
                dim=2,
            )
        return y

    def _forward_train(self, x):
        """训练模式：双分支原始输出 dict（o2o 特征 detach，与 Detect 同口径）"""
        o2m = self._forward_head_pose(x, self.cv2, self.cv3, self.cv4, self.cv4_kpts, self.cv4_sigma)
        x_detached = [feat.detach() for feat in x]
        o2o = self._forward_head_pose(x_detached, self.one2one_cv2, self.one2one_cv3,
                                      self.one2one_cv4, self.one2one_cv4_kpts,
                                      self.one2one_cv4_sigma)
        return {"one2many": o2m, "one2one": o2o}

    def kpts_decode(self, kpts, anchors, strides):
        """关键点解码：(t + anchor) * stride；v 通道 sigmoid（官方 Pose26 口径）

        Args:
            kpts: (B, nk, N) 原始关键点（x,y,v 交错）
            anchors: (2, N) make_anchors 的格点中心（未乘 stride）
            strides: (1, N) 逐点 stride
        """
        ndim = self.kpt_shape[1]
        bs = kpts.shape[0]
        if self.export:
            # 官方 export 分支：view 成 (B,K,D,N)，避免 in-place slice 在 ONNX 图里变成
            # 不可控的 CopySlices（NCNN/部分后端兼容性）
            y = kpts.view(bs, *self.kpt_shape, -1)
            # anchors 已是 (2, N)；不要再 .T 再 reshape（转置视图的 reshape 会按物理序重排，
            # 把 x/y 交错成垃圾——实测 ONNX 坐标翻倍）。直接 reshape 成 (1,1,2,N) 广播。
            a = (y[:, :, :2] + anchors.reshape(1, 1, 2, -1)) * strides.reshape(1, 1, 1, -1)
            if ndim == 3:
                a = torch.cat((a, y[:, :, 2:3].sigmoid()), 2)
            return a.reshape(bs, self.nk, -1)
        y = kpts.clone()
        if ndim == 3:
            y[:, 2::ndim] = y[:, 2::ndim].sigmoid()
        y[:, 0::ndim] = (y[:, 0::ndim] + anchors[0]) * strides
        y[:, 1::ndim] = (y[:, 1::ndim] + anchors[1]) * strides
        return y

    def _decode_pose(self, y, feats):
        """原始分支 -> 已解码像素框 + sigmoid 分数 + 解码关键点（官方 `_inference` 口径）

        o2m 分支官方输出 **xywh**（`decode_bboxes(xywh=not end2end)`），E2E 分支才是 xyxy；
        两者在本文件里分别走 `_decode_pose` / `_e2e_postprocess_pose`，不要混用。
        """
        anchors, strides = make_anchors(feats, self.stride, 0.5)
        boxes = dist2bbox(y["boxes"], anchors.unsqueeze(0), xywh=True) * strides
        scores = y["scores"].sigmoid()
        kpts = self.kpts_decode(y["kpts"], anchors, strides)
        return torch.cat((boxes, scores, kpts), 1)  # (B, 4+nc+nk, N)

    def _e2e_postprocess_pose(self, y, feats):
        """E2E：框 + 关键点共享同一次两阶段 top-k（e2e_topk 的 extra）"""
        anchors, strides = make_anchors(feats, self.stride, 0.5)
        boxes = dist2bbox(y["boxes"], anchors.unsqueeze(0)) * strides
        boxes = boxes.permute(0, 2, 1)  # (B, N, 4)
        scores = y["scores"].permute(0, 2, 1).sigmoid()  # (B, N, nc)
        kpts = self.kpts_decode(y["kpts"], anchors, strides).permute(0, 2, 1)  # (B, N, nk)
        return e2e_topk(boxes, scores, self.max_det, kpts)  # (B, k, 6+nk)

    def postprocess_val(self, preds, feats):
        """验证输出契约：每图 (M, 6+nk) numpy（letterbox 像素）"""
        return list(self._e2e_postprocess_pose(preds["one2one"], feats).detach().cpu().numpy())
