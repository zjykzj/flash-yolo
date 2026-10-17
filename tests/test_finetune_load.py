"""微调初始化：O365（365 类）权重 -> COCO（80 类）头重建时的交集加载"""

import pytest
import torch

from model.build import build_model
from model.weights import load_weights_finetune, save_weights


def _save(model, path):
    save_weights(model, path)
    return path


def test_nc_change_loads_backbone_and_skips_only_head(tmp_path):
    """365 -> 80：backbone/proto/cv2 逐位加载，cv3/semseg 等类别头跳过"""
    src = build_model("yolo26-seg", "n", 640, nc=365)
    path = _save(src, tmp_path / "o365.safetensors")

    dst = build_model("yolo26-seg", "n", 640, nc=80)
    loaded, skipped = load_weights_finetune(dst, path)

    assert skipped, "365 -> 80 应有类别相关头被跳过"
    assert all(k.startswith("model.23.") for k, _, _ in skipped), skipped[:5]
    assert len(loaded) + len(skipped) == len(src.state_dict())
    # 头的形状确实变了，且 backbone 逐位一致
    assert any(k.startswith("model.23.cv3.") for k, _, _ in skipped)
    assert any(k.startswith("model.23.proto.semseg.2") for k, _, _ in skipped)
    assert torch.equal(dst.model[0].conv.weight, src.model[0].conv.weight)
    assert torch.equal(dst.model[-1].proto.cv1.conv.weight, src.model[-1].proto.cv1.conv.weight)


def test_same_nc_is_a_full_load(tmp_path):
    src = build_model("yolo26-seg", "n", 640, nc=80)
    path = _save(src, tmp_path / "same.safetensors")
    dst = build_model("yolo26-seg", "n", 640, nc=80)
    loaded, skipped = load_weights_finetune(dst, path)
    assert not skipped and len(loaded) == len(src.state_dict())
    assert all(torch.equal(v, dst.state_dict()[k]) for k, v in loaded.items())


def test_wrong_scale_is_rejected(tmp_path):
    """不同档位的 key 相同但头外 shape 不同 -> 拒绝，而不是部分加载"""
    path = _save(build_model("yolo26-seg", "n", 640, nc=80), tmp_path / "n.safetensors")
    dst = build_model("yolo26-seg", "s", 640, nc=80)
    with pytest.raises(ValueError, match="outside the task head"):
        load_weights_finetune(dst, path)


def test_wrong_architecture_is_rejected(tmp_path):
    path = _save(build_model("yolo26-seg", "n", 640, nc=80), tmp_path / "seg.safetensors")
    dst = build_model("yolo26", "n", 640, nc=80)
    with pytest.raises(ValueError, match="key mismatch"):
        load_weights_finetune(dst, path)
