"""组合块类：残差/CSP 家族 + 注意力家族

残差与 CSP（backbone/neck 的特征提取主体）：
    Bottleneck   两个 3x3 卷积的残差块
    C3k          C3 变体（CSP 残差块，内部 n 个 Bottleneck）
    C3k2         C2f 变体，按参数三选一：Bottleneck / C3k / Bottleneck+PSABlock

注意力（PSA 系列，backbone 末端与 neck 末块）：
    Attention    多头注意力 + 3x3 depthwise 位置编码
    PSABlock     x = x + attn(x); x = x + ffn(x)
    C2PSA        C2 结构：一半直通、一半过 PSABlock 串

属性树与官方 state_dict 对齐（key 只含属性路径不含类名，见 docs/yolo26-spec.md）。
"""

import torch
import torch.nn as nn

from model.basic.conv import Conv

__all__ = ["Bottleneck", "C3k", "C3k2", "Attention", "PSABlock", "C2PSA"]


class Bottleneck(nn.Module):
    """两个 3x3 卷积的残差块，隐藏通道 c_ = int(c2*e)"""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1, g=g)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3k(nn.Module):
    """C3 变体：两个 1x1 分支，一支过 n 个 3x3 Bottleneck"""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1)
        self.cv2 = Conv(c1, c_, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3k2(nn.Module):
    """C2f 变体：1x1 提升到 2c 后 chunk 两半，一半直通、一半过 n 个块

    块模式（按参数三选一）：
        attn=True  -> Bottleneck + PSABlock（neck 末块）
        c3k=True   -> C3k（backbone 后段与 neck）
        否则        -> Bottleneck（backbone 前段）
    """

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, attn=False, g=1, shortcut=True):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        if attn:
            block = lambda: nn.Sequential(
                Bottleneck(self.c, self.c, shortcut, g),
                PSABlock(self.c, attn_ratio=0.5, num_heads=max(self.c // 64, 1)),
            )
        elif c3k:
            block = lambda: C3k(self.c, self.c, 2, shortcut, g)
        else:
            block = lambda: Bottleneck(self.c, self.c, shortcut, g)
        self.m = nn.ModuleList(block() for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class Attention(nn.Module):
    """多头注意力（PSA 系列内部使用）

    qkv(1x1) -> MHSA -> + pe(v)（3x3 depthwise 位置编码）-> proj(1x1)
    """

    def __init__(self, dim, num_heads=8, attn_ratio=0.5):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim**-0.5
        self.qkv = Conv(dim, dim + 2 * num_heads * self.key_dim, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x):
        b, c, h, w = x.shape
        n = h * w
        qkv = self.qkv(x).reshape(b, self.num_heads, self.key_dim * 2 + self.head_dim, n)
        q, k, v = qkv.split([self.key_dim, self.key_dim, self.head_dim], dim=2)
        attn = ((q * self.scale).transpose(-2, -1) @ k).softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).reshape(b, c, h, w) + self.pe(v.reshape(b, c, h, w))
        return self.proj(x)


class PSABlock(nn.Module):
    """位置敏感注意力块：x = x + attn(x); x = x + ffn(x)"""

    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True):
        super().__init__()
        self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x):
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x


class C2PSA(nn.Module):
    """C2 结构的 PSA：split 一半直通、一半过 n 个 PSABlock"""

    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)
        self.m = nn.Sequential(*(PSABlock(self.c, attn_ratio=0.5, num_heads=max(self.c // 64, 1)) for _ in range(n)))

    def forward(self, x):
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))
