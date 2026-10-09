"""PyTorch 后端：PtEngine（.safetensors -> 仓库模型构建 + strict load）"""

import logging

import torch

from config.inference import IMGSZ
from model.build import build_model
from model.weights import apply_meta, load_meta, load_weights
from utils.engines.base import BaseEngine, resolve_device

__all__ = ["PtEngine"]

logger = logging.getLogger(__name__)


class PtEngine(BaseEngine):
    """PyTorch 权重推理

    imgsz / nc 缺省（None）时从权重 metadata 兜底——训练保存的 best/last 自带
    {nc, imgsz, anchors...}；旧权重没有 metadata 则完全按历史行为（imgsz 640 / 模型 yaml 的 nc）。
    """

    def __init__(self, weights, scale="n", end2end=True, device=None, nc=None, model="yolo26", imgsz=None):
        self.device = resolve_device(device)
        self.scale = scale
        self.arch = model
        self.end2end = end2end
        meta = load_meta(weights)  # 可选 metadata：缺失 = 空 dict
        self.imgsz = int(imgsz) if imgsz is not None else int(meta.get("imgsz") or IMGSZ)
        self.model = build_model(model, scale, imgsz=self.imgsz,
                                 nc=nc if nc is not None else meta.get("nc"))
        for key in apply_meta(self.model, meta):  # v3 anchors 随权重（yaml 为原版也能还原训练值）
            logger.info(f"weights metadata applied: {key}")
        head = self.model.model[-1]
        if hasattr(head, "end2end"):  # V3Detect 无此开关（前向即解码）
            head.end2end = end2end
        load_weights(self.model, weights, strict=True)
        self.model.to(self.device).eval()
        self._warmup()

    def _warmup(self):
        """预热：首次前向触发 CUDA kernel 编译，跑 3 次后计时才反映稳态性能（dummy 用实际输入尺寸）"""
        dummy = torch.zeros(1, 3, self.imgsz, self.imgsz, device=self.device)
        with torch.no_grad():
            for _ in range(3):
                self.model(dummy)

    def _forward(self, img):
        with torch.no_grad():
            return self.model(img.to(self.device))[0].cpu().numpy()

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
