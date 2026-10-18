"""训练中分割验证：低分辨率掩码解码与官方 process_mask(upsample=False) 逐像素一致"""

import numpy as np
import pytest
import torch

from train.validator import VAL_HEADER_SEG, format_val_row
from utils.postprocess import process_mask_lowres

ops = pytest.importorskip("ultralytics.utils.ops")


def test_process_mask_lowres_matches_official():
    torch.manual_seed(0)
    nm, mh, mw = 32, 20, 20
    n = 7
    proto = torch.randn(nm, mh, mw)
    coeff = torch.randn(n, nm)
    boxes = torch.tensor([[5, 5, 60, 60], [10, 15, 55, 70], [0, 0, 40, 40],
                          [30, 30, 80, 80], [12, 8, 70, 50], [2, 2, 78, 78],
                          [20, 20, 60, 60]], dtype=torch.float32)
    shape = (80, 80)  # 网络输入（proto 为其 1/4）
    ref = ops.process_mask(proto, coeff, boxes, shape, upsample=False).numpy()
    ours = process_mask_lowres(proto.numpy(), coeff.numpy(), boxes.numpy(), shape)
    assert ours.shape == ref.shape == (n, mh, mw)
    assert np.array_equal(ours, ref), f"{np.count_nonzero(ours != ref)} differing pixels"


def test_process_mask_lowres_auto_chunk_matches_explicit():
    """自动像素预算分块与任意显式 chunk 逐像素一致（只改执行批次，不改结果）"""
    torch.manual_seed(1)
    proto = torch.randn(32, 20, 20)
    coeff = torch.randn(9, 32)
    boxes = torch.tensor([[0, 0, 40, 40], [5, 5, 70, 70], [12, 8, 60, 50],
                          [3, 3, 78, 78], [20, 20, 79, 79], [1, 1, 30, 30],
                          [10, 20, 50, 60], [25, 5, 75, 55], [6, 6, 66, 66]],
                         dtype=torch.float32)
    auto = process_mask_lowres(proto.numpy(), coeff.numpy(), boxes.numpy(), (80, 80))
    for chunk in (1, 8, 64):
        ref = process_mask_lowres(proto.numpy(), coeff.numpy(), boxes.numpy(), (80, 80), chunk=chunk)
        assert np.array_equal(auto, ref), f"chunk={chunk}"


def test_process_mask_lowres_empty():
    out = process_mask_lowres(np.zeros((32, 20, 20), np.float32), np.zeros((0, 32), np.float32),
                              np.zeros((0, 4), np.float32), (80, 80))
    assert out.shape == (0, 20, 20) and out.dtype == np.uint8


def test_seg_header_and_row():
    assert "mAP50(M)" in VAL_HEADER_SEG and "mAP50-95(M)" in VAL_HEADER_SEG
    assert "P(M)" in VAL_HEADER_SEG and "R(M)" in VAL_HEADER_SEG
    row = format_val_row({"images": 4, "instances": 8, "P": 0.1, "R": 0.2, "mAP@50": 0.3,
                          "mAP@[.5:.95]": 0.4, "P(M)": 0.5, "R(M)": 0.6,
                          "mAP@50(M)": 0.7, "mAP@[.5:.95](M)": 0.8}, seg=True)
    assert "0.8000" in row and row.count("%") == 0
