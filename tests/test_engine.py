"""引擎摘要行与档位推导：pt = 模块树口径、onnx = 部署口径（两者的"参数量"不可混用）"""

from pathlib import Path

import pytest
import torch

from model.weights import scale_from_weights
from utils.engine import device_label, resolve_device

ROOT = Path(__file__).resolve().parent.parent
SAFE_PATH = ROOT / "weights" / "yolo26n.safetensors"
ONNX_PATH = ROOT / "weights" / "yolo26n.onnx"


def test_scale_from_weights():
    """官方命名自带档位；认不出返回 None（调用方报错，而不是闷头按 n 档建模型再在 strict load 处炸）"""
    assert scale_from_weights("weights/yolo26s.safetensors") == "s"
    assert scale_from_weights("weights/yolo26n.onnx") == "n"
    assert scale_from_weights("yolo26x_ep100.safetensors") == "x"
    assert scale_from_weights("YOLO26M.safetensors") == "m"
    assert scale_from_weights("best.safetensors") is None
    assert scale_from_weights("yolo26s2.safetensors") is None, "s 后跟数字不算档位（避免误判 yolo26s2）"
    print("  档位推导：官方命名识别 + 认不出返回 None")


def test_device_label():
    """环境行设备名（engine 构建之前就要打印，故只吃 device 对象/字符串）"""
    assert device_label("cpu") == "CPU"
    assert device_label(resolve_device("cpu")) == "CPU"
    if torch.cuda.is_available():
        assert device_label(resolve_device()).startswith("CUDA "), "未显式指定时应解析到 cuda"
    print("  设备名：CPU / CUDA <型号>")


def test_pt_engine_summary_line():
    """pt 摘要行 = 模块树口径（与训练日志同源，可直接对照 yaml 逐层表）"""
    if not SAFE_PATH.exists():
        pytest.skip("需要 weights/yolo26n.safetensors（先跑 download_weights + convert_weights）")
    from utils.engine import PtEngine

    line = PtEngine(str(SAFE_PATH), device="cpu").summary_line
    assert line == "260 layers · 2,572,280 params", line
    print(f"  pt 摘要行：{line}")


def test_onnx_engine_summary_line():
    """onnx 摘要行 = 部署口径：不打参数量（initializer 含被折叠的 BN，与 pt 不可比）"""
    if not ONNX_PATH.exists():
        pytest.skip("需要 weights/yolo26n.onnx（先跑 scripts/export.py）")
    from utils.engine import OnnxEngine

    line = OnnxEngine(str(ONNX_PATH)).summary_line
    assert line.startswith("ONNX ") and "MiB" in line, line
    assert "in (1, 3, 640, 640)" in line and "out (1, 300, 6)" in line
    assert "params" not in line, "onnx 不得打参数量（2,408,932 initializer ≠ pt 的 2,572,280 parameters）"
    print(f"  onnx 摘要行：{line}")


def test_pt_engine_custom_nc(tmp_path):
    """nc≠80 的权重（自定义数据集训练产物）：引擎按 nc 建头才能 strict 加载

    不给 nc（默认模型 yaml 的 80 类）时由 strict load 以形状不匹配拦下——明确报错，不静默错位。
    """
    from model.weights import save_weights
    from model.build import build_yolo26
    from utils.engine import PtEngine

    path = tmp_path / "nc2.safetensors"
    save_weights(build_yolo26("n", nc=2), path)
    engine = PtEngine(str(path), device="cpu", nc=2)
    assert engine.summary[1] < 2_572_280, "2 类头的参数量应小于 COCO 80 类"
    with pytest.raises(RuntimeError, match="size mismatch"):
        PtEngine(str(path), device="cpu")
    print(f"  nc=2 引擎加载正确（params {engine.summary[1]:,}；缺 nc 时形状不匹配拦下）")
