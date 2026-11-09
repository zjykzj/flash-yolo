"""PyTorch 后端：PtEngine（.safetensors -> 仓库模型构建 + strict load）"""

import logging
from pathlib import Path

import torch

from config.inference import IMGSZ
from model.build import arch_task, build_model
from model.weights import apply_meta, load_meta, load_weights, prompt_mode_from_weights
from utils.engines.base import BaseEngine, resolve_device

__all__ = ["PtEngine"]

logger = logging.getLogger(__name__)


class PtEngine(BaseEngine):
    """PyTorch 权重推理

    imgsz / nc 缺省（None）时从权重 metadata 兜底——训练保存的 best/last 自带
    {nc, imgsz, anchors...}；旧权重没有 metadata 则完全按历史行为（imgsz 640 / 模型 yaml 的 nc）。
    """

    def __init__(self, weights, scale="n", end2end=True, device=None, nc=None, model="yolo26", imgsz=None,
                 agnostic_nms=False):
        self.device = resolve_device(device)
        self.mask_device = self.device  # 掩码解码跟随模型设备（CUDA 下 300 检出 48ms -> 2.6ms）
        self.agnostic_nms = bool(agnostic_nms)  # 仅 NMS 路径生效（E2E 图内无 NMS）
        self.scale = scale
        self.arch = model
        self.task = arch_task(model)
        self.end2end = end2end
        meta = load_meta(weights)  # 可选 metadata：缺失 = 空 dict
        self.imgsz = int(imgsz) if imgsz is not None else int(meta.get("imgsz") or IMGSZ)
        # YOLOE：prompt 模式经 metadata/文件名解析（pf -> lrpc 布局；text -> 文本布局）
        prompt_free = None
        if model == "yoloe-26":
            pm = prompt_mode_from_weights(weights, meta)
            prompt_free = True if pm == "pf" else (False if pm == "text" else None)
            if prompt_free and not end2end:
                raise ValueError("YOLOE prompt-free (lrpc) supports E2E only; use end2end=True "
                                 "(o2m+NMS is available for text-mode weights)")
        # PE 融合产物（fused）：cv3 尾卷积已是 (c3 -> N) 的类锁定卷积，state_dict 里没有
        # cv4/reprta；必须按 struct_nc 重建后再 fuse 出同构架构，strict load 才对得上。
        fused = bool(meta.get("fused"))
        # YOLOE 的结构 nc 以权重 metadata 为准（struct_nc > nc > 显式 nc）：文本布局的 c3
        # 由 checkpoint 结构决定，数据集类数（nc 参数）不能参与建图。其余架构保持原回退。
        if model == "yoloe-26":
            build_nc = meta.get("struct_nc") or meta.get("nc") or nc
        else:
            build_nc = nc if nc is not None else meta.get("nc")
        if model == "yoloe-26" and fused and not meta.get("nc"):
            raise ValueError("fused YOLOE checkpoint metadata lacks `nc`; cannot rebuild the class-locked head")
        self.model = build_model(model, scale, imgsz=self.imgsz, nc=build_nc, prompt_free=prompt_free)
        for key in apply_meta(self.model, meta):  # v3 anchors 随权重（yaml 为原版也能还原训练值）
            logger.info(f"weights metadata applied: {key}")
        head = self.model.model[-1]
        if hasattr(head, "end2end"):  # V3Detect 无此开关（前向即解码）
            head.end2end = end2end
        if fused:
            if getattr(head, "lrpc", None) is not None:
                raise ValueError("fused metadata on a prompt-free (lrpc) checkpoint is invalid")
            self.model.eval()
            head.fuse_text(torch.zeros(1, int(meta["nc"]), head.embed))  # 只重建结构，值由 strict load 覆盖
        load_weights(self.model, weights, strict=True)
        self.model.to(self.device).eval()
        # yoloe 文本布局无 lrpc（pf 布局有）；其余架构恒 False
        self.text_mode = model == "yoloe-26" and getattr(head, "lrpc", None) is None
        self.fused = fused  # PE 融合产物：类锁定，set_classes 会被拒绝
        self.weights_dir = str(Path(weights).resolve().parent)
        self._text_encoder = None  # 惰性构建（set_classes 时）
        self.visual_mode = False  # 当前缓存的是 visual prompt 嵌入（set_visual_*）
        self.visual_names = None
        self.mixed_mode = False  # 文本 + 视觉拼接（官方 get_cls_pe 顺序：文本在前）
        self.prompt_names = meta.get("names") if fused else None  # 类锁定权重自带 prompt 名
        self._warmup()

    def _warmup(self):
        """预热：首次前向触发 CUDA kernel 编译，跑 3 次后计时才反映稳态性能（dummy 用实际输入尺寸）

        文本模式用 dummy tpe（(1,1,512) 零向量）预热前向，之后清空——predict 前
        必须先 set_classes（head 无 tpe 时报错，语义清晰）。fused（PE/类锁定）权重
        不需要也不接受 tpe：它的 cv3 尾卷积直接出 logits，nc 已锁死。
        """
        dummy = torch.zeros(1, 3, self.imgsz, self.imgsz, device=self.device)
        head = self.model.model[-1]
        with torch.no_grad():
            if self.text_mode and not self.fused:
                head.set_tpe(torch.zeros(1, 1, 512, device=self.device))
            for _ in range(3):
                self.model(dummy)
            if self.text_mode and not self.fused:
                head.set_tpe(None)

    def text_embeddings(self, texts, batch=80):
        """类名列表 -> 文本提示嵌入 (1, N, 512)（不缓存；set_classes / 混合 prompt 共用）

        官方 get_text_pe 口径：CLIP tokenize -> MobileCLIP 编码（L2 归一化）-> 按
        batch 分块 -> (1, N, 512) -> 头部 get_tpe（reprta 残差 + L2 归一化）。
        文本编码器权重取模型权重同目录的 mobileclip2_b.safetensors（Apple
        mobileclip2_b.ts 的转换产物），惰性构建并缓存。
        """
        if not self.text_mode:
            raise ValueError("text prompts are only available for yoloe-26 text-mode weights")
        head = self.model.model[-1]
        if getattr(head, "is_fused", False):
            raise ValueError("the model is class-locked (text embeddings fused); "
                             "it rejects new prompts — export a new one for different classes")
        from model.text_prompt import TEXT_ENCODER_FILENAME, encode_text_prompts, load_text_encoder, resolve_text_encoder

        if self._text_encoder is None:
            enc_path = resolve_text_encoder(explicit=Path(self.weights_dir) / TEXT_ENCODER_FILENAME)
            self._text_encoder = load_text_encoder(enc_path, device=self.device)
        return encode_text_prompts(head, list(texts), self._text_encoder, batch=batch)

    def set_classes(self, texts, batch=80):
        """文本模式：类名列表 -> 文本提示嵌入，缓存到 head（纯文本）"""
        tpe = self.text_embeddings(texts, batch)
        self.model.model[-1].set_tpe(tpe)
        self.visual_mode = False
        self.mixed_mode = False
        self.visual_names = None
        self.prompt_names = list(texts)
        return len(texts)

    def visual_embeddings(self, image_bgr, names, prompts_by_class):
        """refer 图 + 每类 bbox/mask -> SAVPE vpe (1, Q, embed)（不缓存；评估侧重排用）"""
        if not self.text_mode:
            raise ValueError("visual prompts are only available for yoloe-26 text-mode weights")
        head = self.model.model[-1]
        if getattr(head, "is_fused", False):
            raise ValueError("the model is class-locked (prompts fused); it rejects visual prompts")
        if not names:
            raise ValueError("visual prompts need at least one class")
        from data.preprocess import preprocess
        from utils.visual_prompt import visual_masks

        tensor, ratio, pad = preprocess(image_bgr, self.imgsz)
        masks = visual_masks(names, prompts_by_class, ratio, pad, self.imgsz,
                             image_bgr.shape[:2]).unsqueeze(0)
        with torch.no_grad():
            feats = self.model.forward_feats(tensor.to(self.device))
            return head.get_vpe(feats, masks.to(self.device))

    def set_visual_prompts(self, image_bgr, names, prompts_by_class):
        """视觉 prompt：refer 图 + 每类 bbox/mask -> SAVPE vpe，缓存到 head

        names/prompts_by_class 的顺序即输出类别顺序（vpe 行序一致）；vpe 已由 SAVPE
        做 L2 归一化，直接作为 ``tpe`` 使用（不过 reprta）。
        """
        vpe = self.visual_embeddings(image_bgr, names, prompts_by_class)
        self.set_visual_embeddings(vpe, names)
        return len(names)

    def set_mixed_prompts(self, texts, image_bgr, visual_names, prompts_by_class):
        """文本 + 视觉混合：按官方 `get_cls_pe` 顺序拼接（文本在前、视觉在后）

        类别名 = texts + visual_names（顺序与嵌入行一致）；同名类别拒绝（避免同一类
        出现两行分类器）。返回总类别数。
        """
        texts, visual_names = list(texts), list(visual_names)
        if not texts or not visual_names:
            raise ValueError("mixed prompts need both text and visual classes")
        dup = sorted(set(texts) & set(visual_names))
        if dup:
            raise ValueError(f"text and visual prompts share class names: {dup[:5]} — "
                             "rename one side or use a single prompt mode")
        tpe = self.text_embeddings(texts)
        vpe = self.visual_embeddings(image_bgr, visual_names, prompts_by_class)
        pe = torch.cat([tpe, vpe], dim=1)
        self.model.model[-1].set_tpe(pe.detach())
        self.visual_mode = False
        self.mixed_mode = True
        self.visual_names = visual_names
        self.prompt_names = texts + visual_names
        return len(self.prompt_names)

    def set_visual_embeddings(self, vpe, names=None):
        """直接缓存已聚合的 vpe (1, N, D)（评估侧 refer 数据集逐类平均后调用）"""
        if not self.text_mode:
            raise ValueError("visual embeddings are only available for yoloe-26 text-mode weights")
        head = self.model.model[-1]
        if getattr(head, "is_fused", False):
            raise ValueError("the model is class-locked (prompts fused); it rejects visual embeddings")
        if vpe.ndim != 3 or vpe.shape[0] != 1:
            raise ValueError(f"visual embeddings must be (1, N, D), got {tuple(vpe.shape)}")
        head.set_tpe(vpe.detach().to(self.device))
        self.visual_mode = True
        self.mixed_mode = False
        self.visual_names = list(names) if names is not None else None
        self.prompt_names = list(names) if names is not None else None
        return vpe.shape[1]

    def _forward(self, img):
        with torch.no_grad():
            out = self.model(img.to(self.device))
        if self.task == "segment":  # 分割头返回 (检测输出, 原型)：剥批后一起交给后处理
            dets, proto = out
            return dets.cpu().numpy()[0], proto.cpu().numpy()[0]
        return out[0].cpu().numpy()

    @property
    def summary(self):
        """(层数, 参数量)，供输出头部摘要；层数 = 叶子模块数（官方 260 层口径）"""
        n_layers = sum(1 for m in self.model.modules() if not list(m.children()))
        return n_layers, sum(p.numel() for p in self.model.parameters())

    @property
    def summary_line(self):
        """模型行的一段：模块树口径（与训练日志同源，可与 yaml 逐层表逐项对照）"""
        n_layers, n_params = self.summary
        return f"{n_layers} layers · {n_params:,} params" if n_layers else f"{n_params:,} params"
