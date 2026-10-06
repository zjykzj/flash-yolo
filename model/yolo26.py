"""YOLO26 模型组装：按 config/yolo26.yaml 构建 nn.Sequential（属性树与官方 state_dict 对齐）"""

import ast
import math
from pathlib import Path

import torch
import torch.nn as nn
import yaml

from model.basic import Conv, C3k2, SPPF, C2PSA
from model.head import Detect

__all__ = ["YOLO26", "build_yolo26"]


class Concat(nn.Module):
    """按通道拼接输入列表（图组装胶水层，非基础算子）"""

    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.cat(x, self.dim)


# 模块注册表（yaml 模块名 -> 类，显式映射避免动态 eval）
MODULES = {
    "Conv": Conv,
    "Concat": Concat,
    "C3k2": C3k2,
    "SPPF": SPPF,
    "C2PSA": C2PSA,
    "Upsample": nn.Upsample,
    "Detect": Detect,
}

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "yolo26.yaml"


def make_divisible(x, divisor=8):
    """向上取整到 divisor 的倍数（通道缩放规则）"""
    return math.ceil(x / divisor) * divisor


class YOLO26(nn.Module):
    """YOLO26 检测模型

    state_dict key = `model.{i}.{属性路径}`，与官方 checkpoint 零映射对齐。
    """

    def __init__(self, cfg_path=None, scale="n"):
        super().__init__()
        cfg_path = cfg_path or CONFIG_PATH
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        self.model, self.save = self._parse(cfg, scale)

    def _parse(self, d, scale):
        """解析 backbone+head 行列表，应用缩放规则（见 docs/yolo26-spec.md 第 2 节）"""
        depth, width, max_ch = d["scales"][scale]
        layers, ch, save = [], [3], set()
        for i, (f, n, name, args) in enumerate(d["backbone"] + d["head"]):
            m = MODULES[name]
            # 字符串参数做 literal 求值（如 "None" -> None；"nearest" 保持字符串）
            raw_args, args = args, []
            for a in raw_args:
                if isinstance(a, str):
                    try:
                        a = ast.literal_eval(a)
                    except (ValueError, SyntaxError):
                        pass
                args.append(a)
            n = max(round(n * depth), 1) if n > 1 else n
            n_repeats = n  # 记录 repeats（打印用；重复模块会把 n 并入参数后重置）
            if name == "Upsample":
                m_ = m(*args)
                c2 = ch[f]
            elif name == "Concat":
                m_ = m(*args)
                c2 = sum(ch[x] for x in f)
                save.update(f)
            elif name == "Detect":
                det_ch = [ch[x] for x in f]
                m_ = m(d["nc"], d["reg_max"], d["end2end"], det_ch)
                m_.stride = torch.tensor([8 * 2**i for i in range(len(det_ch))])
                m_.bias_init()
                c2 = None
                save.update(f)
                args = [d["nc"], d["reg_max"], d["end2end"], det_ch]
            else:
                c1, c2 = ch[f], args[0]
                c2 = make_divisible(min(c2, max_ch) * width)
                args = [c1, c2, *args[1:]]
                if name in ("C3k2", "C2PSA"):  # 重复模块：n 插入参数第 3 位
                    args.insert(2, n)
                    n = 1
                if name == "C3k2" and scale in ("m", "l", "x"):
                    args[3] = True  # m/l/x 档强制 c3k=True
                m_ = m(*args)
            m_.i, m_.f = i, f  # 层索引与来源（forward 路由用）
            # repeats 与构造参数（model/summary.py 打印用）；注意不能占用 m_.n（SPPF 自用该属性）
            m_.repeats, m_.args = n_repeats, args
            layers.append(m_)
            if i == 0:
                ch = []  # 之后 ch[k] = 第 k 层的输出通道
            ch.append(c2)
        return nn.Sequential(*layers), save

    def _forward_layers(self, x, layers):
        """按 from 路由执行层序列；返回 (最终张量, 已保存层的输出表)"""
        y = {}
        for m in layers:
            f = m.f
            if isinstance(f, int):
                if f != -1:
                    x = y[f]
            else:
                x = [x if j == -1 else y[j] for j in f]
            x = m(x)
            if m.i in self.save:
                y[m.i] = x
        return x, y

    def forward(self, x):
        """按 from 路由执行：-1 用当前张量，非负索引取历史层输出，列表取多个层"""
        x, _ = self._forward_layers(x, self.model)
        return x

    def forward_feats(self, x):
        """backbone+neck 前向：返回 head 的输入特征列表

        训练中 val 验证复用同一次前向：特征既供 head 的训练口径双分支输出（算 val 损失），
        又供 E2E 后处理（算指标）——与整模型前向的数值一致（同一路由、同一层序列）。
        """
        x, y = self._forward_layers(x, self.model[:-1])
        f = self.model[-1].f
        if isinstance(f, int):
            return [x] if f == -1 else [y[f]]
        return [x if j == -1 else y[j] for j in f]


def build_yolo26(scale="n", cfg_path=None):
    """构建指定档位模型，默认 eval 且走 E2E 路径"""
    model = YOLO26(cfg_path, scale)
    model.eval()
    head = model.model[-1]
    if isinstance(head, Detect) and getattr(head, "one2one_cv2", None) is not None:
        head.end2end = True
    return model
