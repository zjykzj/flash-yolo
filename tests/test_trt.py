"""TensorRT engine：构建 / 加载 / 与 PyTorch 数值对拍（fp32 raw 输出）。

需要 tensorrt（可选依赖）与 CUDA——缺任一则整文件跳过。
engine 构建约 1 分钟（flash-yolo @320）。
"""

import numpy as np
import pytest
import torch

pytest.importorskip("tensorrt")

if not torch.cuda.is_available():
    pytest.skip("TensorRT engine tests require CUDA", allow_module_level=True)

from model.build import build_model  # noqa: E402
from utils.engine import TRTEngine, build_trt_engine  # noqa: E402


def test_trt_raw_output_matches_pytorch(tmp_path):
    """fp32 engine 的 raw 头输出与 PyTorch 对拍（同权重同输入，稠密张量直接比）"""
    torch.manual_seed(0)
    model = build_model("flash-yolo", None, 320, nc=80)
    model.model[-1].end2end = False  # raw 输出：避开 E2E top-k 的次序/平局问题
    model.eval()
    x = torch.zeros(1, 3, 320, 320)
    with torch.no_grad():
        ref = model(x).numpy()

    onnx_path = tmp_path / "m.onnx"
    with torch.no_grad():
        torch.onnx.export(model, x, str(onnx_path), input_names=["images"], output_names=["output0"],
                          opset_version=18, dynamo=False)
    engine_path = tmp_path / "m.engine"
    build_trt_engine(onnx_path, engine_path, fp16=False)

    eng = TRTEngine(engine_path, model="flash-yolo", end2end=False, imgsz=320)
    out = eng._run(x.numpy())  # (1, 4+nc, N)：与 torch 输出同形（批量维保留，对拍不剥批）
    assert out.shape == ref.shape, (out.shape, ref.shape)
    np.testing.assert_allclose(out, ref, atol=2e-3, rtol=1e-3)
