"""权重 metadata 验收：可选性（缺省/旧文件零影响）+ 三级回退（CLI > metadata > 文件名/默认）"""

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from model.build import build_yolo26, build_yolov3_tiny
from model.weights import apply_meta, load_meta, resolve_arch_scale, resolve_imgsz, save_weights

ROOT = Path(__file__).resolve().parent.parent

CUSTOM_ANCHORS = [[[12, 16], [30, 40], [60, 80]], [[60, 80], [120, 140], [300, 320]]]


def test_save_load_meta_roundtrip(tmp_path):
    """metadata 往返（int/list 类型还原）；无 metadata、未知键、缺文件、非 safetensors 安全降级"""
    m = nn.Linear(3, 2)
    p = tmp_path / "best.safetensors"
    meta = {"arch": "yolov3-tiny", "scale": "tiny", "nc": 2, "imgsz": 512,
            "names": ["a", "b"], "anchors": CUSTOM_ANCHORS}
    save_weights(m, p, metadata=meta)
    assert load_meta(p) == meta

    save_weights(m, tmp_path / "plain.safetensors")  # 旧式保存（无 metadata）
    assert load_meta(tmp_path / "plain.safetensors") == {}
    save_weights(m, tmp_path / "extra.safetensors", metadata={"arch": "yolo26", "custom": "x"})
    assert load_meta(tmp_path / "extra.safetensors") == {"arch": "yolo26"}  # 未知键忽略
    assert load_meta(tmp_path / "missing.safetensors") == {}
    assert load_meta(ROOT / "weights" / "yolo26n.onnx") == {}
    print("  metadata 往返与安全降级正确")


def test_resolve_precedence(tmp_path):
    """三级回退：CLI 显式 > metadata > 文件名/默认（无 metadata 时与历史行为一致）"""
    p = tmp_path / "best.safetensors"  # 文件名推不出任何信息
    save_weights(nn.Linear(1, 1), p, metadata={"arch": "yolo26", "scale": "n", "imgsz": 512})
    assert resolve_arch_scale(str(p)) == ("yolo26", "n")
    assert resolve_imgsz(str(p)) == 512
    assert resolve_arch_scale(str(p), model="yolov3-tiny") == ("yolov3-tiny", None)
    assert resolve_arch_scale(str(p), scale="m") == ("yolo26", "m")
    assert resolve_imgsz(str(p), imgsz=416) == 416

    plain = tmp_path / "yolo26s.safetensors"  # 无 metadata：文件名推断与 640 默认原样保留
    save_weights(nn.Linear(1, 1), plain)
    assert resolve_arch_scale(str(plain)) == ("yolo26", "s")
    assert resolve_imgsz(str(plain)) == 640
    print("  三级回退正确（CLI > metadata > 文件名/默认）")


def test_apply_meta_anchors():
    """v3 anchors 随权重还原；无锚头（yolo26）/无该项无害跳过；形状不符明确报错"""
    model = build_yolov3_tiny(nc=2)
    before = model.model[-1].anchors.clone()
    assert apply_meta(model, {"anchors": CUSTOM_ANCHORS}) == ["anchors"]
    assert torch.equal(model.model[-1].anchors[0], torch.tensor(CUSTOM_ANCHORS[0], dtype=torch.float32))
    assert not torch.equal(before, model.model[-1].anchors)
    assert apply_meta(model, {}) == []
    assert apply_meta(build_yolo26("n"), {"anchors": CUSTOM_ANCHORS}) == []  # 无锚架构忽略
    with pytest.raises(ValueError, match="anchors shape"):
        apply_meta(model, {"anchors": [[[1, 1]]]})
    print("  v3 anchors metadata 应用/跳过/报错正确")


def test_engine_reads_weights_meta(tmp_path):
    """PtEngine 免参数读取 metadata：nc（建头）/ imgsz（前处理+warmup）/ anchors（解码）"""
    from utils.engine import PtEngine

    path = tmp_path / "best.safetensors"  # 文件名推不出档位
    save_weights(build_yolov3_tiny(nc=2), path,
                 metadata={"arch": "yolov3-tiny", "scale": "tiny", "nc": 2, "imgsz": 416,
                           "names": ["a", "b"], "anchors": CUSTOM_ANCHORS})
    assert resolve_arch_scale(str(path)) == ("yolov3-tiny", None)

    engine = PtEngine(str(path), model="yolov3-tiny", scale=None, device="cpu")  # 不给 nc/imgsz
    assert engine.imgsz == 416
    assert torch.equal(engine.model.model[-1].anchors[0], torch.tensor(CUSTOM_ANCHORS[0], dtype=torch.float32))
    dets = engine.predict(np.zeros((480, 640, 3), np.uint8), conf_thres=0.9)
    assert dets.boxes.shape[1] == 4
    print(f"  PtEngine metadata 生效（imgsz {engine.imgsz}，nc 2，自定义 anchors）")
