"""验收1：参数数 / strict load / 与官方推理数值对齐

需要先准备权重（dev-time）：
    python scripts/download_weights.py --model yolo26n
    python scripts/convert_weights.py --src weights/yolo26n.pt --dst weights/yolo26n.safetensors
"""

from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
PT_PATH = ROOT / "weights" / "yolo26n.pt"
SAFE_PATH = ROOT / "weights" / "yolo26n.safetensors"


def _build_ours():
    from model.weights import load_weights
    from model.yolo26 import YOLO26

    model = YOLO26(scale="n").eval()
    load_weights(model, SAFE_PATH, strict=True)
    return model


def test_param_count():
    """结构硬指标：yolo26n = 2,572,280 参数"""
    from model.yolo26 import YOLO26

    model = YOLO26(scale="n")
    assert sum(p.numel() for p in model.parameters()) == 2_572_280


def test_strict_load():
    """strict load 全 key 匹配（加载成功即拓扑一致）"""
    if not SAFE_PATH.exists():
        pytest.skip("需要先运行 scripts/download_weights.py + scripts/convert_weights.py")
    from model.weights import load_weights
    from model.yolo26 import YOLO26

    model = YOLO26(scale="n")
    missing, unexpected = load_weights(model, SAFE_PATH, strict=False)
    assert not missing, f"missing keys: {missing[:5]}"
    assert not unexpected, f"unexpected keys: {unexpected[:5]}"


def test_numeric_alignment():
    """同一输入下与官方推理输出对齐：E2E (1,300,6) 与 NMS 路径解码 (1,84,8400)，误差 < 1e-4"""
    if not (SAFE_PATH.exists() and PT_PATH.exists()):
        pytest.skip("需要官方 .pt 与转换后的 .safetensors")
    ultralytics = pytest.importorskip("ultralytics")

    from utils.postprocess import decode_raw

    torch.manual_seed(0)
    x = torch.randn(1, 3, 640, 640)

    ours = _build_ours()
    official = ultralytics.YOLO(str(PT_PATH)).model.eval()

    with torch.no_grad():
        # ---- E2E 路径（o2o，图内 top-k）----
        official.model[-1].end2end = True
        ref = official(x)
        ref = ref[0] if isinstance(ref, (tuple, list)) else ref  # (1,300,6)
        ours.model[-1].end2end = True
        out = ours(x)
        diff = (ref - out).abs().max().item()
        assert diff < 1e-4, f"E2E 输出误差 {diff:.2e} 超阈值"
        print(f"  E2E 路径误差: {diff:.2e}")

        # ---- NMS 路径（o2m 原始输出 + numpy 解码对齐）----
        official.model[-1].end2end = False
        ref_raw = official(x)
        ref_raw = ref_raw[0] if isinstance(ref_raw, (tuple, list)) else ref_raw  # (1,84,8400) 官方已解码
        ours.model[-1].end2end = False
        out_raw = ours(x)[0].numpy()  # (84,8400) 原始
        boxes, score_map = decode_raw(out_raw)
        decoded = np.concatenate([boxes.T, score_map.T], axis=0)[None]  # (1,84,8400)
        diff_raw = float(np.abs(ref_raw.numpy() - decoded).max())
        assert diff_raw < 1e-4, f"NMS 路径解码误差 {diff_raw:.2e} 超阈值"
        print(f"  NMS 路径解码误差: {diff_raw:.2e}")
