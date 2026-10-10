"""验收2：onnx 与 pt 输出一致

- raw 图：逐元素比较（onnxruntime 与 torch 的实现级浮点噪声，阈值 5e-4）
- E2E 融合图：用真实图片比较；top-k 的平局行（低分背景）在两种实现间顺序可能不同，
  故按置信度过滤后排序逐行比较（有效检测必须完全一致）
"""

import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
SAFE_PATH = ROOT / "weights" / "yolo26n.safetensors"
ONNX_PATH = ROOT / "weights" / "yolo26n.onnx"
RAW_ONNX_PATH = ROOT / "weights" / "yolo26n_raw.onnx"
# 真实测试图片：优先环境变量，其次仓库自带 assets（来源声明见 assets/README.md）
TEST_IMAGE = os.environ.get("FLASH_YOLO_TEST_IMAGE") or ROOT / "assets" / "bus.jpg"


def _export(extra_args, out_path):
    subprocess.run(
        ["python", "scripts/export.py", "--weights", str(SAFE_PATH), "--out", str(out_path), *extra_args],
        cwd=ROOT,
        check=True,
    )


def _build_pt(end2end):
    from model.weights import load_weights
    from model.build import DetectionModel

    model = DetectionModel(scale="n").eval()
    load_weights(model, SAFE_PATH, strict=True)
    model.model[-1].end2end = end2end
    return model


def test_make_anchors_half_dtype_is_export_safe():
    """make_anchors 在 half feats 下返回 half 锚点、且与 fp32 网格逐值相等（fp32 建格再 cast 的回归闸门）。

    动机：half dtype 的 torch.arange 会被 TorchScript ONNX 导出器拒绝（"tensor does not have a
    device"），挡住 model.half() -> fp16 onnx 的整条路；格点值在 fp16 可精确表示，fp32 路径数值不变。
    """
    from utils.anchors import make_anchors

    feats16 = [torch.zeros(1, 4, s, s, dtype=torch.float16) for s in (40, 20, 10)]
    anchors16, strides16 = make_anchors(feats16, [8, 16, 32], 0.5)
    assert anchors16.dtype == torch.float16 and strides16.dtype == torch.float16
    anchors32, strides32 = make_anchors([f.float() for f in feats16], [8, 16, 32], 0.5)
    assert torch.equal(anchors16.float(), anchors32) and torch.equal(strides16.float(), strides32)


def test_export_parity_raw():
    """raw 图：onnx vs pt 逐元素一致"""
    if not SAFE_PATH.exists():
        pytest.skip("需要转换后的权重")
    if not RAW_ONNX_PATH.exists():
        _export(["--raw"], RAW_ONNX_PATH)
    ort = pytest.importorskip("onnxruntime")
    sess = ort.InferenceSession(str(RAW_ONNX_PATH), providers=["CPUExecutionProvider"])

    torch.manual_seed(1)
    x = torch.randn(1, 3, 640, 640)
    model = _build_pt(end2end=False)
    with torch.no_grad():
        ref = model(x).numpy()
    out = sess.run(None, {"images": x.numpy()})[0]
    diff = float(np.abs(ref - out).max())
    assert diff < 5e-4, f"raw: onnx vs pt 误差 {diff:.2e} 超阈值"
    print(f"  raw onnx vs pt 误差: {diff:.2e}")


def test_export_parity_e2e():
    """E2E 融合图：真实图片上有效检测（conf>0.05）排序后逐行一致"""
    if not SAFE_PATH.exists():
        pytest.skip("需要转换后的权重")
    if not Path(TEST_IMAGE).exists():
        pytest.skip(f"需要真实测试图片（可设 FLASH_YOLO_TEST_IMAGE），候选: {TEST_IMAGE}")
    if not ONNX_PATH.exists():
        _export([], ONNX_PATH)
    ort = pytest.importorskip("onnxruntime")
    sess = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])

    import cv2

    from data.preprocess import preprocess

    img = cv2.imread(str(TEST_IMAGE))
    tensor, _, _ = preprocess(img)

    model = _build_pt(end2end=True)
    with torch.no_grad():
        ref = model(tensor).numpy()[0]
    out = sess.run(None, {"images": tensor.numpy()})[0][0]

    r = ref[ref[:, 4] > 0.05]
    o = out[out[:, 4] > 0.05]
    assert len(r) == len(o), f"有效检测数不一致: pt {len(r)} vs onnx {len(o)}"
    # 按 (x1, y1, 分数) 排序后逐行比较
    r = r[np.lexsort((r[:, 0], r[:, 1], -r[:, 4]))]
    o = o[np.lexsort((o[:, 0], o[:, 1], -o[:, 4]))]
    diff = float(np.abs(r - o).max())
    assert diff < 1e-3, f"E2E: onnx vs pt 误差 {diff:.2e} 超阈值"
    print(f"  E2E onnx vs pt 误差: {diff:.2e}（{len(r)} 个有效检测）")
