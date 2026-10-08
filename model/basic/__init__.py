"""基础算子层（model/basic）

按算子类型分文件，无任务语义，每个模块可独立 import、独立单测：

    conv.py   卷积类：autopad / Conv / DWConv / LeakyConv
    pool.py   池化类：SPPF（空间金字塔池化）
    block.py  组合块：Bottleneck / C3k / C3k2 / Attention / PSABlock / C2PSA

属性树与官方 state_dict 对齐（key 只含属性路径不含类名，拆文件不影响权重加载）。
"""

from model.basic.block import Attention, Bottleneck, C2PSA, C3k, C3k2, PSABlock
from model.basic.conv import Conv, DWConv, LeakyConv, autopad
from model.basic.pool import SPPF

__all__ = [
    "autopad",
    "Conv",
    "DWConv",
    "LeakyConv",
    "SPPF",
    "Bottleneck",
    "C3k",
    "C3k2",
    "Attention",
    "PSABlock",
    "C2PSA",
]
