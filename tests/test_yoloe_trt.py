"""YOLOE-26 TensorRT 验收（pf raw / text E2E / text o2m+NMS / fp16）

重（4 次 engine 构建，@320，合计数分钟）——默认跳过，显式打开：
    FLASH_YOLO_TRT=1 pytest tests/test_yoloe_trt.py -v
需要 tensorrt + CUDA + 官方权重（yoloe-26n-seg-pf / yoloe-26n-seg +
mobileclip2_b.safetensors）。

口径：E2E top-k 对 fp32 累加差敏感（与 pt/onnx 的已知差异同源），text E2E 用
conf>0.25 的检出对拍；pf 只有 E2E，改用 raw 稠密输出与 PyTorch 直接对拍（避开 top-k）；
Non-E2E 走 o2m+NMS，检出口径稳定。
"""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
PF_SAFE = ROOT / "weights" / "yoloe-26n-seg-pf.safetensors"
TEXT_SAFE = ROOT / "weights" / "yoloe-26n-seg.safetensors"
ENC = ROOT / "weights" / "mobileclip2_b.safetensors"
TEST_IMAGE = ROOT / "assets" / "traffic.jpg"

pytest.importorskip("tensorrt")
if not torch.cuda.is_available():
    pytest.skip("TensorRT tests require CUDA", allow_module_level=True)
if not os.getenv("FLASH_YOLO_TRT"):
    pytest.skip("set FLASH_YOLO_TRT=1 to run the YOLOE TensorRT acceptance", allow_module_level=True)


def _export(*args):
    res = subprocess.run([sys.executable, "scripts/export.py", *map(str, args)],
                         cwd=ROOT, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr


def _read_image():
    import cv2

    return cv2.imread(str(TEST_IMAGE))


def _assert_dets_close(ref, got, box_tol, score_tol, mask_iou=0.99):
    assert len(ref.boxes) == len(got.boxes), (len(ref.boxes), len(got.boxes))
    assert np.abs(ref.boxes - got.boxes).max() < box_tol
    assert np.abs(ref.scores - got.scores).max() < score_tol
    assert np.array_equal(ref.class_ids, got.class_ids)
    if ref.masks is not None:
        inter = (ref.masks & got.masks).sum((1, 2))
        union = (ref.masks | got.masks).sum((1, 2))
        assert (inter / np.maximum(union, 1)).min() > mask_iou


def test_pf_raw_engine_matches_pytorch(tmp_path):
    """pf 只有 E2E；用 raw 稠密输出建立 TensorRT 与 PyTorch 的数值等价（避开 top-k 平局）"""
    if not PF_SAFE.exists():
        pytest.skip("需要 weights/yoloe-26n-seg-pf.safetensors")
    from model.build import build_model
    from model.weights import apply_meta, load_meta, load_weights
    from utils.engines import TRTEngine, build_trt_engine

    model = build_model("yoloe-26", "n", imgsz=320, prompt_free=True)
    apply_meta(model, load_meta(PF_SAFE))
    load_weights(model, PF_SAFE, strict=True)
    model.eval()
    head = model.model[-1]
    head.end2end = False
    head.export = True  # conf=0 静态：两边同口径的全网格输出
    torch.manual_seed(0)
    x = torch.rand(1, 3, 320, 320)
    with torch.no_grad():
        ref = model(x)[0][0].numpy()  # 剥批：与引擎 _forward 契约一致

    onnx_path = tmp_path / "pf-raw.onnx"
    with torch.no_grad():
        torch.onnx.export(model, x, str(onnx_path), input_names=["images"],
                          output_names=["output0", "output1"], opset_version=18, dynamo=False)
    engine_path = tmp_path / "pf-raw.engine"
    build_trt_engine(onnx_path, engine_path)

    eng = TRTEngine(engine_path, model="yoloe-26", end2end=False, imgsz=320)
    out = eng._forward(x)
    det, proto = out
    assert det.shape == ref.shape, (det.shape, ref.shape)
    # 4585 类 vocab 的 GEMM 累加顺序与 torch 不同：逐元素 max 会被近零值的相对放大主导，
    # 用 p99.9 + 相对最大幅值的界（E2E 部署侧还有 top-k，见文档的敏感性说明）。
    diff = np.abs(det - ref)
    assert np.percentile(diff, 99.9) < 1e-2, float(np.percentile(diff, 99.9))
    assert diff.max() < 0.02 * max(float(np.abs(ref).max()), 1.0), float(diff.max())
    assert proto.shape[0] == 32 and proto.shape[1] == 80 and proto.shape[2] == 80


def test_text_e2e_engine_matches_pt(tmp_path):
    """text 类锁定 E2E：TensorRT 检出与 pt fused 对齐（conf>0.25）"""
    if not TEXT_SAFE.exists() or not ENC.exists() or not TEST_IMAGE.exists():
        pytest.skip("需要 yoloe-26n-seg / mobileclip2_b / traffic.jpg")
    from utils.engines import PtEngine, TRTEngine

    onnx_path = tmp_path / "text-e2e.onnx"
    _export("--weights", TEXT_SAFE, "--classes", "person,bicycle,car",
            "--imgsz", 320, "--out", onnx_path, "--trt")
    engine_path = onnx_path.with_suffix(".engine")
    assert engine_path.exists() and Path(str(engine_path) + ".json").exists()

    img = _read_image()
    pt = PtEngine(str(TEXT_SAFE), model="yoloe-26", scale="n", device="cpu", imgsz=320)
    pt.set_classes(["person", "bicycle", "car"])
    trt = TRTEngine(engine_path, model="yoloe-26", scale="n", imgsz=320)
    assert trt.text_mode and trt.end2end
    _assert_dets_close(pt.predict(img, 0.25), trt.predict(img, 0.25),
                       box_tol=0.1, score_tol=3e-3)


def test_text_non_e2e_engine_matches_pt(tmp_path):
    """text 类锁定 o2m+NMS：TensorRT 检出与 pt 对齐（conf>0.25）"""
    if not TEXT_SAFE.exists() or not ENC.exists() or not TEST_IMAGE.exists():
        pytest.skip("需要 yoloe-26n-seg / mobileclip2_b / traffic.jpg")
    from utils.engines import PtEngine, TRTEngine

    onnx_path = tmp_path / "text-nms.onnx"
    _export("--weights", TEXT_SAFE, "--classes", "person,bicycle,car",
            "--nms", "--imgsz", 320, "--out", onnx_path, "--trt")
    engine_path = onnx_path.with_suffix(".engine")

    img = _read_image()
    pt = PtEngine(str(TEXT_SAFE), model="yoloe-26", scale="n", device="cpu",
                  imgsz=320, end2end=False)
    pt.set_classes(["person", "bicycle", "car"])
    trt = TRTEngine(engine_path, model="yoloe-26", scale="n", imgsz=320, end2end=False)
    assert trt.text_mode and not trt.end2end
    _assert_dets_close(pt.predict(img, 0.25), trt.predict(img, 0.25),
                       box_tol=0.1, score_tol=2e-3)


def test_text_e2e_fp16_engine_matches_pt(tmp_path):
    """fp16 text 类锁定 E2E：半精度引擎与 pt fp32 的检出/掩码容差内一致"""
    if not TEXT_SAFE.exists() or not ENC.exists() or not TEST_IMAGE.exists():
        pytest.skip("需要 yoloe-26n-seg / mobileclip2_b / traffic.jpg")
    from utils.engines import PtEngine, TRTEngine

    onnx_path = tmp_path / "text-e2e16.onnx"
    _export("--weights", TEXT_SAFE, "--classes", "person,bicycle,car",
            "--imgsz", 320, "--out", onnx_path, "--trt", "--fp16")
    engine_path = onnx_path.with_name(onnx_path.stem + ".fp16.engine")
    assert engine_path.exists() and Path(str(engine_path) + ".json").exists()

    img = _read_image()
    pt = PtEngine(str(TEXT_SAFE), model="yoloe-26", scale="n", device="cpu", imgsz=320)
    pt.set_classes(["person", "bicycle", "car"])
    trt = TRTEngine(engine_path, model="yoloe-26", scale="n", imgsz=320)
    assert trt._in.dtype == torch.float16
    _assert_dets_close(pt.predict(img, 0.4), trt.predict(img, 0.4),
                       box_tol=0.5, score_tol=5e-3, mask_iou=0.99)
