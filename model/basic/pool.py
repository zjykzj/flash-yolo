"""池化类模块

YOLO26 没有裸露的池化算子（nn.MaxPool2d 只在 SPPF 内部使用），
池化类在 YOLO26 中的唯一实现是 SPPF（空间金字塔池化块）。
"""

import torch
import torch.nn as nn

from model.basic.conv import Conv

__all__ = ["SPPF"]


class SPPF(nn.Module):
    """空间金字塔池化 - Fast（YOLO26 形式）

    结构：cv1(1x1, 无激活) -> n 次串行 5x5 maxpool -> 4 个中间图拼接 -> cv2(1x1)
    特性：c1==c2 时输出加恒等残差（YOLO26 相对 YOLO11 的新增设计）。

    注：仅实现 yolo26.yaml 使用的 4 参数形式；老式 3 参数 SPPF 不在范围内。
    """

    def __init__(self, c1, c2, k=5, n=3, shortcut=False):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, act=False)
        self.cv2 = Conv(c_ * (n + 1), c2, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.n = n
        self.add = shortcut and c1 == c2

    def forward(self, x):
        y = [self.cv1(x)]
        for _ in range(self.n):
            y.append(self.m(y[-1]))
        y = self.cv2(torch.cat(y, 1))
        return y + x if self.add else y
