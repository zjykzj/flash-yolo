"""权重 metadata 验收：可选性（缺省/旧文件零影响）+ 三级回退（CLI > metadata > 文件名/默认）"""

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from model.build import build_yolo26, build_yolov3_tiny
from model.weights import (apply_meta, load_meta, provenance_meta, resolve_arch_scale, resolve_imgsz,
                           save_weights)

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


def test_provenance_meta(tmp_path):
    """出处：version/repo 常驻；commit/describe best-effort（非仓库不写键，而不是写空串）"""
    from config import __repo__, __version__
    from utils.paths import git_commit, git_describe

    meta = provenance_meta()
    assert meta["version"] == __version__ and meta["repo"] == __repo__
    assert set(meta) <= {"version", "repo", "commit", "describe"}
    assert git_commit(tmp_path) is None and git_describe(tmp_path) is None  # 非 git 目录：None（不写键）
    if commit := git_commit():   # 仓库内：完整 40 位 sha，与 meta.json 的 git.commit 同源
        assert meta["commit"] == commit and len(commit) == 40
    else:                        # 无 git / 无 .git（镜像内）：键缺席
        assert "commit" not in meta
    if describe := git_describe():      # describe 原始输出：tag-距离-g短sha[-dirty]，人读的落差说明
        assert meta["describe"] == describe and describe
        if "-g" in describe:            # tag 之后：尾段是 HEAD 的缩写 sha（-dirty 后缀先摘掉）
            assert commit.startswith(describe.rsplit("-g", 1)[1].removesuffix("-dirty")), describe
        else:                           # 正好在 tag 上（无 tag 时 --always 退化成短 sha）
            assert describe.startswith("v") or commit.startswith(describe), describe
    else:
        assert "describe" not in meta

    save_weights(nn.Linear(1, 1), tmp_path / "prov.safetensors", metadata={**meta, "arch": "yolo26"})
    got = load_meta(tmp_path / "prov.safetensors")
    assert got["repo"] == __repo__ and got["version"] == __version__  # 读取侧白名单常驻
    assert got.get("commit") == meta.get("commit")                   # 缺则两边都没有
    assert got.get("describe") == meta.get("describe")
    print(f"  出处写入/还原正确（commit={'有' if 'commit' in meta else '无'}，describe={meta.get('describe')}）")


def test_engine_reads_weights_meta(tmp_path):
    """PtEngine 免参数读取 metadata：nc（建头）/ imgsz（前处理+warmup）/ anchors（解码）"""
    from utils.engines import PtEngine

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
