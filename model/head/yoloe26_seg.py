"""YOLOE-26 分割头：prompt-free（lrpc）与文本（BNContrastiveHead）双布局

结构 = yolo26-seg 的 Segment26 同族：backbone/neck（0-22 层）与 yolo26-seg 逐行相同，
官方 yoloe-26-seg.yaml 只换最后一行头（`YOLOESegment26 [nc, 32, 256, 512, True]`）。

头内部按 `prompt_free` 构建两种布局（官方 set_vocab 原地重参数化的两个状态）：

    prompt_free=True（官方 -pf 权重，M1 目标）：
        cv2.{i}.{0,1}                box 分支（截断，无最后 Conv2d -> 拆到 lrpc.loc）
        cv3.{i}.{0.0,0.1,1.0,1.1}    嵌入分支（截断，无最后 Conv2d -> 拆到 lrpc.pf）
        lrpc.{i}.{vocab,pf,loc}      vocab=Linear(c3->V) 烘焙词汇；pf=提案过滤；loc=框回归
        cv5.{i} / proto              mask 系数 / Proto26（与 Segment26 同构，逐位复用）
        one2one_*                    end2end 双头副本（one2one_lrpc 同构）
        reprta / savpe               保留（pf 前向不用；strict load 对齐，同 semseg 先例）
    prompt_free=False（文本模式，M2 目标）：
        cv2/cv3 完整（含最后 Conv2d）；cv4 = BNContrastiveHead（BN + 余弦相似度 + logit_scale）

前向契约：
    eval（end2end=True） : (B, max_det, 6+nm) 图内 top-k。max_det=1000 是 YOLOE 口径
                           （官方 pf 头实例实测；yolo26 系为 300）。
    eval（end2end=False）: (B, 4+V+nm, M) 原始输出。text 布局 M=8400（完整网格，支持
                           o2m+NMS，best-class-per-anchor 口径）；pf 布局 M 随提案过滤变，
                           仅支持 E2E（raw/NMS 对 pf 不开放）。
    导出                 : export 标志 -> conf=0 静态路径（全锚点通过、无动态掩码，官方口径）。

文本模式前向（M2）：forward_text = 官方 YOLOEDetect.forward_head 口径——每级
BNContrastiveHead(cv3 区域嵌入, tpe) 出分数，tpe 经 get_tpe（reprta + L2）预处理，
nc 随 prompt 数量动态变。
"""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.basic import Conv, DWConv
from model.head.yolo26_detect import e2e_topk
from model.head.yolo26_seg import Proto26
from utils.anchors import dist2bbox, make_anchors

__all__ = ["BNContrastiveHead", "LRPCHead", "SwiGLUFFN", "Residual", "SAVPE", "YOLOESegment26"]


class BNContrastiveHead(nn.Module):
    """文本-视觉对比头（BN 归一化 + L2 归一化文本嵌入 + 余弦相似度）

    官方口径：区域特征过 BN（非 L2）、文本嵌入 L2 归一化后 einsum 求相似度，乘可学习
    logit_scale（初值 -1.0）+ bias（初值 -10.0，与检测头 cls 先验同量级）。norm 的
    eps=0.001（官方 ckpt 实测；PyTorch 默认 1e-5 会与官方输出差 ~0.35——M1 时 pf
    前向不用 cv4 故未暴露）。
    """

    def __init__(self, embed_dims=512):
        super().__init__()
        self.norm = nn.BatchNorm2d(embed_dims, eps=0.001)
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        self.logit_scale = nn.Parameter(-1.0 * torch.ones([]))

    def fuse(self):
        """导出 fuse（官方 bn_h.fuse()）：文本侧已融进 cv3 尾卷积，本头变恒等"""
        del self.norm
        del self.bias
        del self.logit_scale
        self.forward = self.forward_fuse

    @staticmethod
    def forward_fuse(x, w):
        return x

    def forward(self, x, w):
        """x: (B, C, H, W) 区域特征；w: (B, K, C) 文本嵌入 -> (B, K, H, W) 相似度 logits"""
        x = self.norm(x)
        w = F.normalize(w, dim=-1, p=2)
        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias


def _fuse_conv_bn(conv, bn):
    """Conv2d + BatchNorm2d 折叠（标准口径，同官方 fuse_conv_and_bn）"""
    w = conv.weight
    b = conv.bias
    scale = bn.weight / (bn.running_var + bn.eps).sqrt()
    w_f = w * scale.view(-1, 1, 1, 1)
    b_f = (b - bn.running_mean) * scale + bn.bias
    fused = nn.Conv2d(conv.in_channels, conv.out_channels, 1,
                      device=conv.weight.device, dtype=conv.weight.dtype)
    fused.weight.data.copy_(w_f)
    fused.bias.data.copy_(b_f)
    return fused


class SwiGLUFFN(nn.Module):
    """SwiGLU 前馈网络（reprta 的残差块主体）：Linear(512->2048) -> silu 门控 -> Linear(1024->512)"""

    def __init__(self, gc, ec, e=4):
        super().__init__()
        self.w12 = nn.Linear(gc, e * ec)
        self.w3 = nn.Linear(e * ec // 2, ec)

    def forward(self, x):
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        return self.w3(F.silu(x1) * x2)


class Residual(nn.Module):
    """残差包装（reprta = Residual(SwiGLUFFN)）：w3 权重与偏置零初始化（官方口径）"""

    def __init__(self, m):
        super().__init__()
        self.m = m
        nn.init.zeros_(self.m.w3.bias)
        nn.init.zeros_(self.m.w3.weight)

    def forward(self, x):
        return x + self.m(x)


class SAVPE(nn.Module):
    """空间感知视觉 prompt 编码器（SAVE 模块）

    官方逐层口径：三尺度特征经 cv1/cv2 归一化到 c3 后 concat，cv3 出嵌入、cv4 出
    softmax 注意力分数；视觉 prompt 掩码经 cv5 与注意力特征 concat 过 cv6，与嵌入
    按掩码加权聚合（非掩码位置置 -inf 走 softmax），输出 L2 归一化 (B, Q, embed)。
    两种布局都构建（pf 前向不用）；前向 M3（视觉 prompt）启用。
    """

    def __init__(self, ch, c3, embed):
        super().__init__()
        self.cv1 = nn.ModuleList(
            nn.Sequential(
                Conv(x, c3, 3), Conv(c3, c3, 3),
                nn.Upsample(scale_factor=i * 2) if i in (1, 2) else nn.Identity(),
            )
            for i, x in enumerate(ch)
        )
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c3, 1), nn.Upsample(scale_factor=i * 2) if i in (1, 2) else nn.Identity())
            for i, x in enumerate(ch)
        )
        self.c = 16
        self.cv3 = nn.Conv2d(3 * c3, embed, 1)
        self.cv4 = nn.Conv2d(3 * c3, self.c, 3, padding=1)
        self.cv5 = nn.Conv2d(1, self.c, 3, padding=1)
        self.cv6 = nn.Sequential(Conv(2 * self.c, self.c, 3), nn.Conv2d(self.c, self.c, 3, padding=1))

    def forward(self, x, vp):
        """x: 三尺度 neck 特征；vp: (B, Q, H, W) 视觉 prompt 掩码（P3 分辨率）-> (B, Q, embed)"""
        y = [self.cv2[i](xi) for i, xi in enumerate(x)]
        y = self.cv4(torch.cat(y, dim=1))

        x = [self.cv1[i](xi) for i, xi in enumerate(x)]
        x = self.cv3(torch.cat(x, dim=1))

        B, C, H, W = x.shape
        Q = vp.shape[1]

        x = x.view(B, C, -1)

        y = y.reshape(B, 1, self.c, H, W).expand(-1, Q, -1, -1, -1).reshape(B * Q, self.c, H, W)
        vp = vp.reshape(B, Q, 1, H, W).reshape(B * Q, 1, H, W)

        y = self.cv6(torch.cat((y, self.cv5(vp)), dim=1))

        y = y.reshape(B, Q, self.c, -1)
        vp = vp.reshape(B, Q, 1, -1)

        score = y * vp + torch.logical_not(vp) * torch.finfo(y.dtype).min
        score = F.softmax(score, dim=-1).to(y.dtype)
        aggregated = score.transpose(-2, -3) @ x.reshape(B, self.c, C // self.c, -1).transpose(-1, -2)

        return F.normalize(aggregated.transpose(-2, -3).reshape(B, Q, -1), dim=-1, p=2)


class LRPCHead(nn.Module):
    """pf 头单元：vocab（烘焙词汇）+ pf（提案过滤）+ loc（框回归）

    官方 set_vocab 把文本嵌入融合进 cv3 末卷积后原地重参数化成此结构：pf / loc 分别是
    cv3 / cv2 的末层卷积（Conv2d），vocab 是 (c3 -> V) 的映射。pf 输出 embed=512 通道、
    **只取第 0 通道**作提案分数（官方 forward 口径）。

    enabled 是官方发布权重的结构事实：`enabled=i != 2` —— **P5（level 2）不做 conv2linear
    转换、不做提案过滤**（vocab 保持 4D Conv2d，全锚点通过，掩码全 True；官方 lrpc.2.
    vocab.weight 是 (V, c3, 1, 1) 而非 (V, c3)）。必须原样对齐，strict load 与逐位输出
    都依赖它。conf=0（导出静态路径）时 enabled 级全锚点通过、无掩码；否则逐图 sigmoid
    阈值过滤 + 批内并集掩码（任一图保留的锚点整批保留），被逐图过滤的锚点分数填 -inf。
    """

    def __init__(self, vocab, pf, loc, enabled=True):
        super().__init__()
        self.vocab = vocab
        self.pf = pf
        self.loc = loc
        self.enabled = enabled

    def forward(self, cls_feat, loc_feat, conf):
        """cls_feat: (B, c3, H, W)；loc_feat: (B, c2, H, W)

        返回 (loc 原始输出 (B, 4, H, W), 分数 (B, V, M), 保留掩码或 None)。
        """
        if not self.enabled:  # P5 级：vocab 保持 Conv2d（4D 进出），全锚点通过
            cls_feat = self.vocab(cls_feat)
            loc_feat = self.loc(loc_feat)
            return (loc_feat, cls_feat.flatten(2),
                    cls_feat.new_ones(cls_feat.shape[2] * cls_feat.shape[3], dtype=torch.bool))
        if not conf:  # 静态导出：全锚点通过
            cls_feat = self.vocab(cls_feat.flatten(2).transpose(-1, -2))
            return self.loc(loc_feat), cls_feat.transpose(-1, -2), None
        keep = self.pf(cls_feat)[:, 0].flatten(1).sigmoid() > conf  # (B, N) 逐图提案
        mask = keep.any(0)  # 批内并集
        cls_feat = self.vocab(cls_feat.flatten(2).transpose(-1, -2)[:, mask])
        cls_feat = cls_feat.masked_fill(~keep[:, mask, None], float("-inf"))
        return self.loc(loc_feat), cls_feat.transpose(-1, -2), mask


class YOLOESegment26(nn.Module):
    """YOLOE-26 分割头（官方 YOLOESegment26 口径：嵌入头 + Proto26）

    与 yolo26_seg.Segment26 的关系：cv2（box）/ cv5（mask 系数）/ proto（Proto26）逐位
    同构；cv3 换成 512-d 嵌入分支；新增 cv4 / reprta / savpe（文本布局）或 lrpc（pf 布局）。
    活 nc 语义：pf 下 = 词汇数 V（官方 set_vocab 后口径，E2E 类别索引按它算）；文本下随
    每次前向的 prompt 数量变（M2）。c3 隐藏通道按 **yaml nc** 算（vocab 的输入维度），
    与 V 无关（n 档 80 / s 档 128，已实测官方同值）。
    """

    conf = 0.001    # pf 提案过滤阈值（官方 YOLOEDetect 类默认）
    export = False  # 导出标志（scripts/export.py 设置）：conf=0 静态路径
    dynamic = False

    def __init__(self, nc=80, nm=32, npr=256, embed=512, reg_max=1, end2end=True, ch=(),
                 prompt_free=False, vocab_size=4585, pf_channels=512):
        super().__init__()
        # max_det 随官方资产而异（实测 ckpt）：pf 资产 1000（官方 pf 发布流程设过）、
        # 文本资产 300（Detect 类默认）；yolo26 系同为 300
        self.max_det = 1000 if prompt_free else 300
        self.nl = len(ch)
        self.reg_max = reg_max
        self.nm = nm
        self.npr = npr
        self.embed = embed
        self.nc = vocab_size if prompt_free else nc
        self.no = self.nc + reg_max * 4
        self.stride = torch.zeros(self.nl)
        c2 = max(16, ch[0] // 4, reg_max * 4)  # box 分支隐藏通道（n 档 = 16）
        c3 = max(ch[0], min(nc, 100))          # 嵌入分支隐藏通道（n 档 80 / s 档 128）
        c5 = max(ch[0] // 4, nm)               # mask 系数隐藏通道

        self.cv2 = nn.ModuleList(
            nn.Sequential(
                Conv(x, c2, 3), Conv(c2, c2, 3),
                *([] if prompt_free else [nn.Conv2d(c2, 4 * reg_max, 1)]),
            )
            for x in ch
        )
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                *([] if prompt_free else [nn.Conv2d(c3, embed, 1)]),
            )
            for x in ch
        )
        if not prompt_free:
            self.cv4 = nn.ModuleList(BNContrastiveHead(embed) for _ in ch)
        self.cv5 = nn.ModuleList(
            nn.Sequential(Conv(x, c5, 3), Conv(c5, c5, 3), nn.Conv2d(c5, nm, 1)) for x in ch
        )
        if prompt_free:
            # 官方 set_vocab 口径：enabled=i != 2——P5 级 vocab 保持 Conv2d、不做提案过滤。
            # pf 输出通道随官方资产版本而异（n=512 复用 cv3 尾卷积 / s=1 专用单通道），
            # 由 yaml 档位数据传入（pf_channels）；行为一致：只取第 0 通道作提案分数。
            self.lrpc = nn.ModuleList(
                LRPCHead(
                    nn.Conv2d(c3, vocab_size, 1) if i == 2 else nn.Linear(c3, vocab_size),
                    nn.Conv2d(c3, pf_channels, 1),
                    nn.Conv2d(c2, 4 * reg_max, 1),
                    enabled=i != 2,
                )
                for i in range(len(ch))
            )
        if end2end:
            self.one2one_cv2 = copy.deepcopy(self.cv2)
            self.one2one_cv3 = copy.deepcopy(self.cv3)
            self.one2one_cv5 = copy.deepcopy(self.cv5)
            if not prompt_free:
                self.one2one_cv4 = copy.deepcopy(self.cv4)
            else:
                self.one2one_lrpc = copy.deepcopy(self.lrpc)
        self.reprta = Residual(SwiGLUFFN(embed, embed))
        self.savpe = SAVPE(ch, c3, embed)
        self.proto = Proto26(ch, npr, nm, nc)
        self._end2end = False

    # ---- 推理模式选择（与 yolo26 Detect 同语义：engine 构建后置 True 走 E2E）----
    @property
    def end2end(self):
        return getattr(self, "one2one_cv2", None) is not None and (self._end2end or self.cv2 is None)

    @end2end.setter
    def end2end(self, value):
        self._end2end = value

    # ---- 前向 ----
    def forward(self, x, tpe=None):
        if self.training and getattr(self, "one2one_cv2", None) is not None:
            return self._forward_train(x, tpe)
        if getattr(self, "lrpc", None) is not None:
            y = self.forward_lrpc(x)
        else:
            if tpe is None:
                tpe = getattr(self, "_tpe", None)
            if tpe is None and not getattr(self, "is_fused", False):
                raise ValueError(
                    "YOLOE text mode requires tpe (text prompt embeddings) — "
                    "call set_classes() on the engine first")
            y = self.forward_text(x, tpe)
        proto = self.proto(x, return_semantic=False)
        return y, proto

    def _resolve_tpe(self, tpe):
        """训练/验证共用的 prompt 解析：显式 tpe > 缓存 _tpe；fused 后无需 tpe"""
        if tpe is None:
            tpe = getattr(self, "_tpe", None)
        if tpe is None and not getattr(self, "is_fused", False):
            raise ValueError(
                "YOLOE text-mode training requires tpe (text prompt embeddings) — "
                "call set_classes()/set_tpe() before training")
        return tpe

    def _forward_head_train(self, x, box_head, cls_head, cv4_head, cv5_head, tpe):
        """训练口径单分支原始输出（scores 为 logits，boxes 为 ltrb 网格单位）

        fused 后 cv3 尾卷积直接输出类别 logits（cv4 恒等）——传 cv4_head=None 跳过；
        未 fused 时走 BNContrastiveHead(cv3 区域嵌入, tpe)，与 forward_text 同口径但
        不 sigmoid、不做 E2E 解码（损失/分配器消费原始输出）。
        """
        bs = x[0].shape[0]
        boxes = torch.cat(
            [box_head[i](x[i]).reshape(bs, self.reg_max * 4, -1) for i in range(self.nl)], dim=-1)
        if cv4_head is None:
            scores = torch.cat(
                [cls_head[i](x[i]).reshape(bs, self.nc, -1) for i in range(self.nl)], dim=-1)
        else:
            if tpe.shape[0] != bs:
                tpe = tpe.expand(bs, -1, -1)
            scores = torch.cat(
                [cv4_head[i](cls_head[i](x[i]), tpe).reshape(bs, self.nc, -1)
                 for i in range(self.nl)], dim=-1)
        coeff = torch.cat(
            [cv5_head[i](x[i]).reshape(bs, self.nm, -1) for i in range(self.nl)], dim=-1)
        return {"boxes": boxes, "scores": scores, "mask_coefficient": coeff}

    def _forward_train(self, x, tpe=None):
        """训练模式：双分支原始输出 + 原型（契约同 Segment26._forward_train）

        o2o 分支吃 detached 特征（梯度只经 o2m 回 o2m 头/backbone）；o2o 的 proto 同样
        detach。pf 布局的训练尚未接入（词汇烘焙在 lrpc 中，无文本/PE 路径）——显式报错，
        不静默走错分支。fused 文本布局（PE 线性探测的产物）cv4 已折叠进 cv3，tpe 可省。
        """
        if getattr(self, "lrpc", None) is not None:
            raise NotImplementedError(
                "YOLOE prompt-free layout does not support training; "
                "train the text layout (prompt_free=false) instead")
        tpe = self._resolve_tpe(tpe)
        fused = getattr(self, "is_fused", False)
        if not fused:
            self.nc = tpe.shape[1]
        cv4_o2m = None if fused else self.cv4
        cv4_o2o = None if fused else self.one2one_cv4
        o2m = self._forward_head_train(x, self.cv2, self.cv3, cv4_o2m, self.cv5, tpe)
        x_detached = [feat.detach() for feat in x]
        o2o = self._forward_head_train(
            x_detached, self.one2one_cv2, self.one2one_cv3, cv4_o2o, self.one2one_cv5, tpe)
        proto = self.proto(x, return_semantic=False)
        o2m["proto"], o2o["proto"] = proto, proto.detach()
        return {"one2many": o2m, "one2one": o2o}

    def _e2e_postprocess(self, y, feats):
        """E2E 解码（训练验证与 postprocess_val 共用）：ltrb -> xyxy -> 两阶段 top-k"""
        anchors, strides = make_anchors(feats, self.stride, 0.5)
        boxes = dist2bbox(y["boxes"], anchors.unsqueeze(0)) * strides
        boxes = boxes.permute(0, 2, 1)
        scores = y["scores"].permute(0, 2, 1).sigmoid()
        extra = y["mask_coefficient"].permute(0, 2, 1) if "mask_coefficient" in y else None
        return e2e_topk(boxes, scores, self.max_det, extra)

    def postprocess_val(self, preds, feats):
        """验证输出契约：每图 (M, 6+nm) [x1,y1,x2,y2,conf,cls,系数...] numpy"""
        return list(self._e2e_postprocess(preds["one2one"], feats).detach().cpu().numpy())

    def set_tpe(self, tpe):
        """缓存文本提示嵌入（引擎 set_classes 调用）；前向未显式传 tpe 时使用"""
        self._tpe = tpe

    def fuse_text(self, tpe):
        """导出 fuse（官方 fuse/_fuse_tp 口径）：文本嵌入融合进 cv3 尾卷积，类锁定

        tpe: (1, N, 512) get_tpe 输出（已 reprta + L2——官方 fuse 的输入即此）。
        每级每分支：cv4 的 BN 折叠进 cv3 尾卷积 -> 尾卷积换成 (c3 -> N)（权重 =
        (tpe×exp(logit_scale)) @ 折叠权重），cv4 变恒等、reprta 删掉变 Identity。
        之后前向输出即类别 logits，不接受新 prompt（set_classes 守卫报错）。
        """
        assert not self.training
        tpe = tpe.squeeze(0)  # (N, 512)
        branches = [(self.cv3, self.cv4)]
        if getattr(self, "one2one_cv3", None) is not None:
            branches.append((self.one2one_cv3, self.one2one_cv4))
        for cv3h, cv4h in branches:
            for i in range(self.nl):
                bn = cv4h[i].norm
                conv = cv3h[i][-1]
                t = tpe * cv4h[i].logit_scale.exp()
                fused = _fuse_conv_bn(conv, bn)
                w = fused.weight.squeeze(-1).squeeze(-1)  # (embed, c3)
                b = fused.bias  # (embed,)
                new = nn.Conv2d(w.shape[1], t.shape[0], 1,
                                device=w.device, dtype=w.dtype).requires_grad_(False)
                new.weight.data.copy_((t @ w).unsqueeze(-1).unsqueeze(-1))
                new.bias.data.copy_(t @ b + cv4h[i].bias)
                cv3h[i][-1] = new
                cv4h[i].fuse()
        del self.reprta
        self.reprta = nn.Identity()
        self.nc = tpe.shape[0]  # fuse 后类别数 = 锁定的 prompt 数（前向不再依赖 tpe）
        self.is_fused = True

    def get_tpe(self, tpe):
        """文本提示嵌入的头部变换（官方 get_tpe 口径）：reprta 残差 + L2 归一化"""
        return None if tpe is None else F.normalize(self.reprta(tpe), dim=-1, p=2)

    def get_vpe(self, x, vp):
        """视觉提示嵌入（官方 get_vpe 口径）：多尺度特征 + 1/8 掩码 -> SAVPE 聚合

        x 是 head 的输入特征列表，vp 是 (B, Q, H/8, W/8) 掩码；返回 (B, Q, embed)
        （SAVPE 已做 L2 归一化，前向时直接作为 tpe 使用，不再过 reprta）。
        """
        if vp.shape[1] == 0:  # 无 visual prompt：返回空嵌入（官方同口径）
            return torch.zeros(x[0].shape[0], 0, self.embed, device=x[0].device, dtype=x[0].dtype)
        if vp.ndim == 4:
            vp = self.savpe(x, vp)
        if vp.ndim != 3:
            raise ValueError(f"visual prompt embeddings must be (B, Q, D), got {tuple(vp.shape)}")
        return vp

    def forward_text(self, x, tpe):
        """文本前向（官方 YOLOEDetect.forward_head 口径）

        分数 = BNContrastiveHead(cv3 区域嵌入, tpe)：BN 归一化区域特征与 L2 归一化
        文本嵌入求余弦相似度，乘 exp(logit_scale) + bias。nc 随 prompt 数量动态变
        （self.nc = tpe.shape[1]；fuse 后 cv4 是恒等、tpe 可为 None）。tpe 形状
        (1|B, N, 512)——官方 pipeline 产出 (1,N,512)、前向时 expand 到 batch。
        框/mask 系数与 pf 路径同构。E2E top-k 复用同一实现。
        """
        bs = x[0].shape[0]
        if tpe is not None:
            self.nc = tpe.shape[1]
            if tpe.shape[0] != bs:
                tpe = tpe.expand(bs, -1, -1)
        cv2 = self.one2one_cv2 if self.end2end else self.cv2
        cv3 = self.one2one_cv3 if self.end2end else self.cv3
        cv4 = self.one2one_cv4 if self.end2end and getattr(self, "one2one_cv4", None) is not None else self.cv4
        cv5 = self.one2one_cv5 if self.end2end else self.cv5
        boxes, scores = [], []
        for i in range(self.nl):
            boxes.append(cv2[i](x[i]).view(bs, self.reg_max * 4, -1))
            scores.append(cv4[i](cv3[i](x[i]), tpe).reshape(bs, self.nc, -1))
        anchors, strides = make_anchors(x, self.stride, 0.5)
        dbox = dist2bbox(torch.cat(boxes, 2), anchors.unsqueeze(0)) * strides
        mc = torch.cat([cv5[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2)
        # 官方口径：o2m/E2E 都输出 sigmoid 分数（官方 _inference 已做 sigmoid），框已解码
        y = torch.cat((dbox, torch.cat(scores, 2).sigmoid(), mc), 1)
        if self.end2end:
            y = e2e_topk(y[:, :4].permute(0, 2, 1), y[:, 4:4 + self.nc].permute(0, 2, 1),
                         self.max_det, extra=y[:, 4 + self.nc:].permute(0, 2, 1))
        return y

    def forward_lrpc(self, x):
        """pf 前向：提案过滤 -> 全网格解码 -> 按 index 取回 -> (E2E) 图内 top-k

        与官方 forward_lrpc 逐行对齐：cv2/cv3/lrpc 按 end2end 选分支；conf 提案过滤的掩码
        是批内并集（被逐图过滤的锚点填 -inf）；锚点网格始终是**完整特征图**——解码在全网格
        上做、再按 index 取回（loc 输出不裁剪）。E2E top-k 复用 yolo26_detect.e2e_topk
        （两阶段选择唯一实现）。返回 (B, max_det, 6+nm) 或原始 (B, 4+V+nm, M)。
        """
        bs = x[0].shape[0]
        cv2 = self.one2one_cv2 if self.end2end else self.cv2
        cv3 = self.one2one_cv3 if self.end2end else self.cv3
        cv5 = self.one2one_cv5 if self.end2end else self.cv5
        lrpc = self.one2one_lrpc if self.end2end and getattr(self, "one2one_lrpc", None) is not None else self.lrpc
        conf = 0 if self.export and not self.dynamic else self.conf
        boxes, scores, index = [], [], []
        for i in range(self.nl):
            cls_feat = cv3[i](x[i])
            loc_feat = cv2[i](x[i])
            box, score, idx = lrpc[i](cls_feat, loc_feat, conf)
            boxes.append(box.view(bs, self.reg_max * 4, -1))
            scores.append(score)
            index.append(idx)
        index = torch.cat(index) if conf else None
        anchors, strides = make_anchors(x, self.stride, 0.5)
        dbox = dist2bbox(torch.cat(boxes, 2), anchors.unsqueeze(0)) * strides  # (B, 4, N) xyxy
        dbox = dbox if index is None else dbox[..., index]
        mc = torch.cat([cv5[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2)
        mc = mc if index is None else mc[..., index]
        y = torch.cat((dbox, torch.cat(scores, 2).sigmoid(), mc), 1)  # (B, 4+V+nm, M)
        if self.end2end:
            y = e2e_topk(y[:, :4].permute(0, 2, 1), y[:, 4:4 + self.nc].permute(0, 2, 1),
                         self.max_det, extra=y[:, 4 + self.nc:].permute(0, 2, 1))
        return y
