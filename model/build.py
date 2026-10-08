"""模型组装：按 config/models/*.yaml 构建 nn.Sequential 模块树（属性树与官方 state_dict 对齐）

行语法 [from, repeats, module, args]；MODULES 是唯一模块注册表（显式映射，避免动态 eval）。
"""

import ast
import math
from pathlib import Path

import torch
import torch.nn as nn
import yaml

from model.basic import C2PSA, C3k2, Conv, LeakyConv, SPPF
from model.head import Detect
from model.head_v3 import V3Detect

__all__ = ["DetectionModel", "build_model", "build_yolo26", "build_yolov3_tiny",
           "arch_display_name", "ARCHS", "YOLO26_CONFIG_PATH", "V3_CONFIG_PATH"]


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
    "MaxPool2d": nn.MaxPool2d,
    "ZeroPad2d": nn.ZeroPad2d,
    "LeakyConv": LeakyConv,
    "Detect": Detect,
    "V3Detect": V3Detect,
}

# 通道保持型算子（args[0] 不是输出通道）：上采样 / 下采样池化 / padding 胶水层
CHANNEL_PRESERVING = {"Upsample", "MaxPool2d", "ZeroPad2d"}

YOLO26_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "models" / "yolo26.yaml"
V3_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "models" / "yolov3-tiny.yaml"

# 架构注册表：name -> yaml 路径 + 默认档位 + 展示名模板（新增架构在此登记）
ARCHS = {
    "yolo26": {"cfg": YOLO26_CONFIG_PATH, "default_scale": "n", "display": "YOLO26{scale}"},
    "yolov3-tiny": {"cfg": V3_CONFIG_PATH, "default_scale": "tiny", "display": "YOLOv3-tiny"},
}


def arch_display_name(arch, scale):
    """汇总行展示名（trainer 启动块与 model/summary 共用）"""
    return ARCHS[arch]["display"].format(scale=scale)


def make_divisible(x, divisor=8):
    """向上取整到 divisor 的倍数（通道缩放规则）"""
    return math.ceil(x / divisor) * divisor


class DetectionModel(nn.Module):
    """yaml 组装出的检测模型

    state_dict key = `model.{i}.{属性路径}`，与官方 checkpoint 零映射对齐。
    """

    def __init__(self, cfg_path=None, scale="n", imgsz=640, nc=None):
        super().__init__()
        cfg_path = cfg_path or YOLO26_CONFIG_PATH
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        self.model, self.save = self._parse(cfg, scale, imgsz, nc)

    def _parse(self, d, scale, imgsz=640, nc=None):
        """解析 backbone+head 行列表，应用缩放规则（depth · width · max_channels，源自论文）

        imgsz、nc 是训练侧的数据事实（先验尺寸 / 数据集类别数），默认 None/640 保留
        yaml 里的结构值（推理/导出与官方权重对齐走默认即可）。
        """
        if scale not in d["scales"]:
            raise ValueError(f"scale {scale!r} not in {sorted(d['scales'])}")
        depth, width, max_ch = d["scales"][scale]
        rows = d["backbone"] + d["head"]
        # 路由归一化（parse 期做一次，forward 只认 -1 与绝对索引）：相对负数 -k -> i-k；
        # 同时前移登记 save——凡被引用的层（不限于 Concat/Detect 输入）forward 时都按其暂存
        norm_f, save = [], set()
        for i, (f, _n, _name, _args) in enumerate(rows):
            f = [i + j if j < -1 else j for j in f] if isinstance(f, (list, tuple)) else (i + f if f < -1 else f)
            for j in (f if isinstance(f, (list, tuple)) else [f]):
                if j == -1:  # -1 = 当前张量，无需暂存
                    continue
                if j < -1 or j >= i:
                    raise ValueError(f"layer {i}: route {j} unresolvable")
                save.add(j)
            norm_f.append(f)
        layers, ch = [], [3]
        for i, ((_, n, name, args), f) in enumerate(zip(rows, norm_f)):
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
            if name in CHANNEL_PRESERVING:
                m_ = m(*args)
                c2 = ch[f]
            elif name == "Concat":
                m_ = m(*args)
                c2 = sum(ch[x] for x in f)
            elif name == "Detect":
                det_ch = [ch[x] for x in f]
                nc_ = d["nc"] if nc is None else nc  # 数据集类别数覆盖 yaml 结构值（训练侧）
                m_ = m(nc_, d["reg_max"], d["end2end"], det_ch)
                m_.stride = torch.tensor([8 * 2**i for i in range(len(det_ch))])
                m_.bias_init(imgsz)
                c2 = None
                args = [nc_, d["reg_max"], d["end2end"], det_ch]
            elif name == "V3Detect":
                det_ch = [ch[x] for x in f]
                nc_ = d["nc"] if nc is None else nc
                anchors, strides = d["anchors"], d["strides"]
                if not len(anchors) == len(strides) == len(det_ch):
                    raise ValueError(f"anchors/strides/from length mismatch: "
                                     f"{len(anchors)}/{len(strides)}/{len(det_ch)}")
                m_ = m(nc_, det_ch, anchors, strides, d.get("ref_imgsz", 416), imgsz)
                c2 = None
                args = [nc_, det_ch]
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


def build_model(arch="yolo26", scale=None, imgsz=640, nc=None, cfg_path=None):
    """按架构注册表构建模型（原始构造：不 eval、不改 end2end 状态）

    scale=None 用架构默认档位（yolo26: n / yolov3-tiny: tiny）；cfg_path 显式覆盖 yaml。
    """
    spec = ARCHS.get(arch)
    if spec is None:
        raise ValueError(f"unknown arch {arch!r}; available: {sorted(ARCHS)}")
    return DetectionModel(cfg_path or spec["cfg"], scale or spec["default_scale"], imgsz, nc)


def build_yolo26(scale="n", cfg_path=None, imgsz=640, nc=None):
    """构建指定档位模型，默认 eval 且走 E2E 路径（imgsz/nc 只影响从零训练与自定义类别数）"""
    model = build_model("yolo26", scale, imgsz, nc, cfg_path)
    model.eval()
    head = model.model[-1]
    if isinstance(head, Detect) and getattr(head, "one2one_cv2", None) is not None:
        head.end2end = True
    return model


def build_yolov3_tiny(nc=None, imgsz=640, cfg_path=None):
    """构建 YOLOv3-tiny（eval 模式；前向即完成解码，无 end2end 开关）"""
    model = build_model("yolov3-tiny", imgsz=imgsz, nc=nc, cfg_path=cfg_path)
    model.eval()
    return model
