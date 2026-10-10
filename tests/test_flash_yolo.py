"""flash-yolo（本项目自有架构）：构建 / E2E 前向 / 算力预算冒烟

算力预算是核心设计承诺（硬指标：GFLOPs 必须确实低于 yolo26n）——此处做回归闸门。
"""

from pathlib import Path

import pytest
import torch

from model.build import DetectionModel, build_model
from model.summary import profile_flops

ROOT = Path(__file__).resolve().parent.parent


def test_e2e_forward_shape():
    """E2E 前向与 yolo26 同契约（Detect 头复用）：显式置位 end2end 后输出 (1, 300, 6)

    置位约定与 build_yolo26 / PtEngine / Trainer val 相同——原始 build_model 构造不带。
    """
    model = build_model("flash-yolo", None, 320, nc=80)
    model.model[-1].end2end = True
    model.eval()
    with torch.no_grad():
        out = model(torch.zeros(1, 3, 320, 320))
    assert tuple(out.shape) == (1, 300, 6)


def test_raw_forward_shape():
    """默认（未置位 end2end）：o2m 原始输出 (1, 4+nc, N)，N = Σ(S//stride)²"""
    model = build_model("flash-yolo", None, 320, nc=80)
    model.eval()
    with torch.no_grad():
        out = model(torch.zeros(1, 3, 320, 320))
    assert tuple(out.shape) == (1, 84, 2100)  # 84 = 4 + 80；2100 = 40² + 20² + 10²


def test_gflops_budget_below_yolo26n():
    """@640 的 GFLOPs 必须低于 yolo26n（实测 −31%；闸门放宽到 −20%）"""
    fl = lambda arch, scale: profile_flops(build_model(arch, scale, 640, nc=80), 640) / 1e9  # noqa: E731
    assert fl("flash-yolo", None) < fl("yolo26", "n") * 0.8


@pytest.mark.parametrize("yaml_name", ["flash-yolo.yaml", "flash-yolo-s1.yaml", "flash-yolo-s2.yaml",
                                       "flash-yolo-s3.yaml", "flash-yolo-s4.yaml"])
def test_flash_yolo_variant_yamls_build(yaml_name):
    """config/models/ 下全部 flash-yolo 结构（现役 + s1-s4 筛选语料）都能组装 + 前向（yaml 回归闸门）"""
    model = DetectionModel(cfg_path=ROOT / "config" / "models" / yaml_name, scale="flash", imgsz=320)
    model.eval()
    with torch.no_grad():
        out = model(torch.zeros(1, 3, 320, 320))
    assert tuple(out.shape) == (1, 84, 2100)  # 与 flash-yolo 同拓扑（P3/P4/P5：40² + 20² + 10²）
