"""YOLOv3-tiny I/O 验收：权重名解析 / darknet 转换器往返与尺寸守卫 / pt·onnx 引擎与导出"""

import struct
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from model.build import build_model, build_yolov3_tiny
from model.weights import load_weights, resolve_arch_scale, save_weights

ROOT = Path(__file__).resolve().parent.parent


def test_resolve_arch_scale():
    """(arch, scale) 解析：显式参数优先，缺省从文件名推断；v3 档位固定"""
    assert resolve_arch_scale("weights/yolov3-tiny.safetensors") == ("yolov3-tiny", None)
    assert resolve_arch_scale("weights/yolov3-tiny_ep100.safetensors") == ("yolov3-tiny", None)
    assert resolve_arch_scale("weights/yolov3-tiny.onnx") == ("yolov3-tiny", None)
    assert resolve_arch_scale("weights/yolo26s.safetensors") == ("yolo26", "s")
    assert resolve_arch_scale("best.safetensors") == (None, None)
    assert resolve_arch_scale("best.safetensors", model="yolov3-tiny") == ("yolov3-tiny", None)
    assert resolve_arch_scale("best.safetensors", scale="m") == ("yolo26", "m")
    assert resolve_arch_scale("yolov3-tiny.safetensors", scale="s") == ("yolov3-tiny", None)
    print("  架构/档位解析正确")


def _synth_darknet(src, header_ints=4):
    """按 darknet 存储序合成随机权重文件（层序取自转换器；序本身的正确性由真实权重对拍验证，
    此处覆盖解析/映射/键集/头部判别等机制）"""
    from scripts.convert_weights import _darknet_specs

    model = build_model("yolov3-tiny", nc=80)
    specs = _darknet_specs(model)
    ref, chunks = {}, []
    torch.manual_seed(0)
    for module, kind in specs:
        conv = module.conv if hasattr(module, "conv") else module
        w = torch.randn(*conv.weight.shape) * 0.1
        if kind == "bn":
            beta, gamma = torch.randn(conv.out_channels), torch.rand(conv.out_channels) + 0.5
            mean, var = torch.randn(conv.out_channels), torch.rand(conv.out_channels) + 0.5
            chunks += [beta, gamma, mean, var, w]
            ref[id(module)] = (kind, beta, gamma, mean, var, w)
        else:
            b = torch.randn(conv.out_channels)
            chunks += [b, w]
            ref[id(module)] = (kind, b, w)
    blob = torch.cat([c.reshape(-1) for c in chunks]).numpy().astype("<f4").tobytes()
    src.write_bytes(struct.pack(f"<{header_ints}i", *([0, 2, 0, 32013312, 114514][:header_ints])) + blob)
    return model, specs, ref


@pytest.mark.parametrize("header_ints", [4, 5])
def test_darknet_converter_roundtrip(tmp_path, header_ints):
    """合成 .weights（4/5 int 头都兼容）-> 转换 -> 严格加载，逐张量与生成值一致"""
    from scripts.convert_weights import _convert_darknet

    src = tmp_path / "yolov3-tiny.weights"
    model, specs, ref = _synth_darknet(src, header_ints)
    dst = tmp_path / "yolov3-tiny.safetensors"
    _convert_darknet(src, dst, nc=80)

    missing, unexpected = load_weights(model, dst, strict=True)
    assert not missing and not unexpected
    names = {id(m): n for n, m in model.named_modules()}
    sd = model.state_dict()
    for module, kind in specs:
        prefix = names[id(module)]
        if kind == "bn":
            _, beta, gamma, mean, var, w = ref[id(module)]
            assert torch.equal(sd[f"{prefix}.conv.weight"], w)
            assert torch.equal(sd[f"{prefix}.bn.bias"], beta)
            assert torch.equal(sd[f"{prefix}.bn.weight"], gamma)
            assert torch.equal(sd[f"{prefix}.bn.running_mean"], mean)
            assert torch.equal(sd[f"{prefix}.bn.running_var"], var)
        else:
            _, b, w = ref[id(module)]
            assert torch.equal(sd[f"{prefix}.weight"], w)
            assert torch.equal(sd[f"{prefix}.bias"], b)
    print(f"  darknet 转换往返一致（{header_ints} int 头，{len(specs)} 个块）")


def test_darknet_converter_size_guard(tmp_path):
    """文件长度不匹配（多/少字节）报错而非静默错位"""
    from scripts.convert_weights import _convert_darknet

    src = tmp_path / "bad.weights"
    _synth_darknet(src, header_ints=4)
    src.write_bytes(src.read_bytes() + b"\x00" * 8)  # 多 2 个 float：既非 16 也非 20 字节头部
    with pytest.raises(ValueError, match="does not match"):
        _convert_darknet(src, tmp_path / "x.safetensors", nc=80)
    print("  尺寸守卫正确")


def test_pt_engine_v3(tmp_path):
    """PtEngine 走 v3 分支：解码 + 按类 NMS，Detections 形状正确"""
    from utils.engine import PtEngine

    path = tmp_path / "yolov3-tiny.safetensors"
    save_weights(build_yolov3_tiny(nc=2), path)
    engine = PtEngine(str(path), model="yolov3-tiny", scale=None, nc=2, device="cpu")
    img = np.zeros((480, 640, 3), np.uint8)
    dets = engine.predict(img, conf_thres=0.5)
    assert dets.boxes.shape[1] == 4 and len(dets.boxes) == len(dets.scores) == len(dets.class_ids)
    assert "layers" in engine.summary_line
    print(f"  v3 引擎推理 OK（{engine.summary_line}，检出 {len(dets.scores)}）")


def test_v3_export_parity(tmp_path):
    """scripts/export.py --model yolov3-tiny：单输出 (1, 6000, 5+nc)，onnx vs pt 数值一致"""
    ort = pytest.importorskip("onnxruntime")

    path = tmp_path / "yolov3-tiny.safetensors"
    model = build_yolov3_tiny(nc=2)
    save_weights(model, path)
    onnx_path = tmp_path / "v3.onnx"
    subprocess.run(["python", "scripts/export.py", "--weights", str(path), "--out", str(onnx_path),
                    "--model", "yolov3-tiny", "--nc", "2"], cwd=ROOT, check=True)

    from utils.engine import OnnxEngine

    line = OnnxEngine(str(onnx_path), model="yolov3-tiny").summary_line
    assert "out (1, 6000, 7)" in line, line

    torch.manual_seed(0)
    x = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        ref = model(x).numpy()
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    out = sess.run(None, {"images": x.numpy()})[0]
    # 融合 BN 与 torch 分离计算的实现级浮点差经 exp 放大在像素量级框坐标上：
    # 以 imgsz 归一后与 raw 图的 5e-4 口径同尺度（实测 ≤3e-5）
    diff = float(np.abs(ref - out).max())
    assert diff / 640 < 5e-4, f"v3 onnx vs pt 误差 {diff:.2e}（归一 {diff / 640:.2e}）"
    print(f"  v3 导出/数值一致（误差 {diff:.2e}，归一 {diff / 640:.2e}，输出 {out.shape}）")
