"""模型结构打印：逐层参数表 + 整体 summary（参考 yolov5 models/yolo.py 的打印格式）

用法:
    python model/summary.py                     # 打印 yolo26n
    python model/summary.py --scale s           # yolo26 其他档位
    python model/summary.py --model yolov3-tiny # 其他架构

FLOPs 用 torch 内置 FlopCounterMode 统计（纯网络，不含 E2E 图内 top-k），零额外依赖。
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # 仓库根目录入 sys.path（直接 python model/summary.py 运行时需要）

import torch

from config import __version__
from model.build import ARCHS, arch_display_name, build_model

__all__ = ["model_summary", "profile_flops"]


def layer_rows(model):
    """逐层信息: [(idx, from, repeats, params, 模块全名, 构造参数)]"""
    rows = []
    for m in model.model:
        n_params = sum(p.numel() for p in m.parameters())
        name = f"{type(m).__module__}.{type(m).__name__}"
        rows.append((m.i, m.f, m.repeats, n_params, name, m.args))
    return rows


def profile_flops(model, imgsz=640, batch=1, device=None):
    """统计纯网络 FLOPs（临时切到 raw 头输出，排除 E2E 图内 top-k）

    device: 统计张量的设备（None = CPU；模型在 GPU 上时必须传入对应设备）。
    注意：本函数会切 model.eval() 且不恢复训练模式，调用方需自行恢复。
    """
    from torch.utils.flop_counter import FlopCounterMode

    head = model.model[-1]
    saved = getattr(head, "end2end", None)
    if saved is not None:
        head.end2end = False
    x = torch.zeros(batch, 3, imgsz, imgsz, device=device)
    model.eval()
    with torch.no_grad():
        with FlopCounterMode(display=False) as fcm:
            model(x)
    if saved is not None:
        head.end2end = saved
    return fcm.get_total_flops()


def summary_lines(model, imgsz=640, batch=1, device=None, name="YOLO26"):
    """逐层表 + 汇总行（字符串列表，trainer 启动块复用），返回 (lines, 层数, 参数量, GFLOPs)

    name: 汇总行的模型名（trainer 传 YOLO26{scale}，独立工具用通用 YOLO26）
    """
    n_layers = sum(1 for m in model.modules() if not list(m.children()))
    n_params = sum(p.numel() for p in model.parameters())
    gflops = profile_flops(model, imgsz, batch, device=device) / 1e9

    lines = [f"{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}"]
    for idx, f, n, layer_params, name_, args in layer_rows(model):
        lines.append(f"{idx:>3}{str(f):>18}{n:>3}{layer_params:>10}  {name_:<40}{str(args):<30}")
    lines.append(f"{name} summary: {n_layers} layers, {n_params:,} parameters, {gflops:.1f} GFLOPs")
    return lines, n_layers, n_params, gflops


def model_summary(model, imgsz=640, batch=1, name="YOLO26"):
    """打印逐层表与整体 summary，返回 (层数, 参数量, FLOPs)"""
    lines, n_layers, n_params, gflops = summary_lines(model, imgsz, batch, name=name)
    for line in lines:
        print(line)
    return n_layers, n_params, gflops


def main():
    parser = argparse.ArgumentParser(description="print model architecture")
    parser.add_argument("--model", default="yolo26", choices=sorted(ARCHS), help="architecture name")
    parser.add_argument("--scale", default=None, help="model scale (yolo26: n/s/m/l/x；缺省用架构默认档)")
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    device = f"CUDA {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "CPU"
    spec = ARCHS[args.model]
    scale = args.scale or spec["default_scale"]
    print(f"model/summary.py: cfg={spec['cfg']}, scale={scale}, imgsz={args.imgsz}")
    print(f"Flash-YOLO {__version__} 🚀 Python {sys.version.split()[0]} · torch {torch.__version__} · {device}\n")

    model = build_model(args.model, args.scale, args.imgsz)
    model_summary(model, imgsz=args.imgsz, name=arch_display_name(args.model, scale))


if __name__ == "__main__":
    main()
