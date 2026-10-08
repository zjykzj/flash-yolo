"""卷积类算子

- Conv: Conv2d(bias=False) + BN + SiLU，YOLO26 几乎所有特征变换的基础算子
- DWConv: 深度可分离卷积（groups=gcd(c1,c2)），仅用于 Detect 的 cls 分支
- 激活说明: YOLO26 无自定义激活算子，SiLU 直接用 nn.SiLU（见 Conv.act），不单设 act 模块
"""

import math

import torch.nn as nn

__all__ = ["autopad", "Conv", "DWConv"]


def autopad(k, p=None):
    """same 填充：默认 kernel//2"""
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    """Conv2d(bias=False) + BN + SiLU

    注意 BN 的 eps=0.001（非常规值，与官方实现一致；数值对齐必须相同）。
    属性树 conv/bn/act。
    """

    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        # eps=0.001 与官方实现一致（数值对齐必须；推理路径由它决定）。
        # momentum=0.03：官方 yaml 构建的模型全部 114 个 BN 均为 0.03（训练时
        # running stats 更新速率；只影响验证/EMA 侧，不影响 train 前向的 batch stats）
        self.bn = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DWConv(Conv):
    """深度可分离卷积：groups = gcd(c1, c2)"""

    def __init__(self, c1, c2, k=1, s=1, act=True):
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), act=act)
