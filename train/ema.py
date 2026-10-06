"""模型 EMA：权重指数滑动平均（验证与保存均用 EMA 副本）"""

import copy
import math

import torch

__all__ = ["ModelEMA"]


class ModelEMA:
    """EMA 副本（fp32 deepcopy + eval）

    decay 爬升（官方 torch_utils.ModelEMA 口径）: d = decay * (1 - exp(-steps/tau))
    ——早期小 d 快速跟随，约 5τ 步后逼近 decay（tau=2000 时 100 轮约为 0.9999 的
    稳态平滑；此前用 (1+s)/(tau+s) 公式，早期一样快但永远到不了 decay，100 轮时
    仅 0.989≈百步窗，见 CHANGELOG）。滑动对象含浮点 buffer（BN running stats 一并平均）。
    """

    def __init__(self, model, decay=0.9999, tau=2000):
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = decay
        self.tau = tau
        self.steps = 0

    @torch.no_grad()
    def update(self, model):
        self.steps += 1
        d = self.decay * (1.0 - math.exp(-self.steps / self.tau))
        v_es, v_ms = [], []
        for (k_e, v_e), (k_m, v_m) in zip(self.ema.state_dict().items(), model.state_dict().items()):
            assert k_e == k_m, f"EMA key 不一致: {k_e} vs {k_m}"
            if v_m.dtype.is_floating_point:
                v_es.append(v_e)
                v_ms.append(v_m)
        if v_es:  # foreach 逐元素：与逐张量 mul_/add_ 位级一致，kernel 数降一个量级
            torch._foreach_mul_(v_es, d)
            torch._foreach_add_(v_es, v_ms, alpha=1 - d)

    def eval_model(self):
        """返回 EMA 副本（已 eval；调用方负责设 head.end2end 等推理标志）"""
        return self.ema
