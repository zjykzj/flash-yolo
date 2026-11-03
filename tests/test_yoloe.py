"""YOLOE-26 验收（M1，prompt-free）：结构 / 转换 metadata / strict load / 官方逐位对齐 / 导出对拍

需要先准备权重（dev-time；官方 .pt 从 ultralytics assets 下载）：
    python scripts/convert_weights.py --src weights/yoloe-26n-seg-pf.pt --dst weights/yoloe-26n-seg-pf.safetensors

结构事实（已实测）：官方 -pf 资产由不同版本代码产出——n 档（2026-09）pf 提案卷积 512 通道
（复用 cv3 尾卷积）、s 档（2026-01）1 通道（专用单通道）；P5（level 2）vocab 保持 4D Conv2d、
不做提案过滤（enabled=i != 2）。官方 yaml 注释的参数量（5,615,540）是陈旧值，不含
savpe/reprta——本测试断言的 7,837,694 是 strict-load 一致的实值。
"""

import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
PF_PT = ROOT / "weights" / "yoloe-26n-seg-pf.pt"
PF_SAFE = ROOT / "weights" / "yoloe-26n-seg-pf.safetensors"
PF_ONNX = ROOT / "weights" / "yoloe-26n-seg-pf.onnx"
TEXT_PT = ROOT / "weights" / "yoloe-26n-seg.pt"
TEST_IMAGE = ROOT / "assets" / "traffic.jpg"

N_PF_PARAMS = 7_837_694  # yoloe-26n-seg-pf parameters()（见文件头说明）


def _build(prompt_free, scale="n"):
    from model.build import build_model

    return build_model("yoloe-26", scale, prompt_free=prompt_free)


def _official_keys(src):
    """官方 ckpt 的 (key -> shape)（反序列化需 ultralytics dev 依赖）"""
    pytest.importorskip("ultralytics")
    model = torch.load(src, map_location="cpu", weights_only=False)["model"]
    return {k: tuple(v.shape) for k, v in model.float().state_dict().items()}


def test_param_count_pf():
    """结构硬指标：yoloe-26n-seg-pf = 7,837,694 参数（strict-load 一致口径）"""
    assert sum(p.numel() for p in _build(True).parameters()) == N_PF_PARAMS


def test_head_build_key_sets_pf():
    """构建产物 key 集与官方 pf ckpt 完全一致（含 P5 级 vocab 保持 Conv2d 的结构怪癖）"""
    if not PF_PT.exists():
        pytest.skip("需要官方权重 weights/yoloe-26n-seg-pf.pt")
    ref = _official_keys(PF_PT)
    mine = {k: tuple(v.shape) for k, v in _build(True).state_dict().items()}
    assert set(mine) == set(ref)
    assert all(mine[k] == ref[k] for k in ref)
    # 官方 set_vocab 口径的回归闸门：P5（level 2）vocab 是 4D Conv2d、enabled=False；其余级 Linear
    assert ref["model.23.lrpc.2.vocab.weight"] == (4585, 80, 1, 1)
    assert ref["model.23.lrpc.0.vocab.weight"] == (4585, 80)


def test_head_build_key_sets_text():
    """文本布局构建（M1 只建结构，前向 M2）：key 集与官方文本 ckpt 一致"""
    if not TEXT_PT.exists():
        pytest.skip("需要官方权重 weights/yoloe-26n-seg.pt")
    ref = _official_keys(TEXT_PT)
    mine = {k: tuple(v.shape) for k, v in _build(False).state_dict().items()}
    assert set(mine) == set(ref)
    assert all(mine[k] == ref[k] for k in ref)
    head = _build(False).model[-1]  # 文本前向是 M2 范围：直接调头应明确报 NotImplementedError
    feats = [torch.zeros(1, c, 80 // 2**i, 80 // 2**i) for i, c in enumerate((64, 128, 256))]
    with pytest.raises(NotImplementedError):
        head.eval()(feats)


def test_convert_pf_metadata():
    """转换器：prompt_mode/pf_channels/nc/names 随权重走

    nc 写 yaml 结构值（80）而非 pf 活 nc（4585 = 烘焙词汇数）；names 是 4585 条 RAM++ 词汇。
    """
    if not PF_PT.exists():
        pytest.skip("需要官方权重 weights/yoloe-26n-seg-pf.pt")
    if not PF_SAFE.exists():
        subprocess.run(["python", "scripts/convert_weights.py",
                        "--src", str(PF_PT), "--dst", str(PF_SAFE)], cwd=ROOT, check=True)
    from model.weights import load_meta

    meta = load_meta(PF_SAFE)
    assert meta["arch"] == "yoloe-26"
    assert meta["prompt_mode"] == "pf"
    assert meta["nc"] == 80
    assert meta["pf_channels"] == 512
    assert len(meta["names"]) == 4585


def test_strict_load_and_forward_parity():
    """strict load 全 key + 与官方 fp32 前向逐位一致（E2E 检测 + 原型；zeros/random 双输入）"""
    if not PF_PT.exists() or not PF_SAFE.exists():
        pytest.skip("需要官方权重与转换产物")
    pytest.importorskip("ultralytics")
    from model.weights import apply_meta, load_meta, load_weights

    m = _build(True)
    apply_meta(m, load_meta(PF_SAFE))
    missing, unexpected = load_weights(m, PF_SAFE, strict=True)
    assert not missing and not unexpected
    m.eval()
    m.model[-1].end2end = True
    ref = torch.load(PF_PT, map_location="cpu", weights_only=False)["model"].float().eval()
    ref.model[-1].end2end = True
    for x in (torch.zeros(1, 3, 640, 640), torch.rand(1, 3, 640, 640)):
        with torch.no_grad():
            y_o, p_o = ref(x)[0]  # 官方返回 ((y, proto), preds)
            y_m, p_m = m(x)
        assert torch.equal(y_m, y_o), "E2E 检测输出与官方逐位一致"
        assert torch.equal(p_m, p_o), "原型与官方逐位一致"


def test_vocab_map():
    """names -> vocab 映射：只保留数据集类别列；缺失类别报错；同义词回退"""
    from eval.coco_evaluator import build_vocab_map

    m = build_vocab_map(["person", "car", "bus"], ["bus", "person"])
    assert m == {2: 0, 0: 1}
    # 同义词：数据集名不在词汇时经 vocab_synonyms 回退（COCO "airplane" -> RAM++ "plane"）
    m = build_vocab_map(["person", "car", "plane"], ["airplane", "person"], synonyms={"airplane": "plane"})
    assert m == {2: 0, 0: 1}
    with pytest.raises(ValueError):
        build_vocab_map(["person"], ["person", "bicycle"])


def test_export_parity():
    """导出图（conf=0 静态口径，官方同款）vs pt 同路径：conf>0.25 的有效检测完全一致

    4585 类的近并列分数只在低分尾部出现，由 torch/ORT 后端各自决胜——口径：conf>0.25 的
    检测按 (x1, y1, -score) 排序后逐行比较，框/分数/系数 < 1e-3 且零类别翻转（实测 65/65）。
    """
    ort = pytest.importorskip("onnxruntime")
    if not PF_SAFE.exists() or not TEST_IMAGE.exists():
        pytest.skip("需要转换后的权重与测试图片")
    if not PF_ONNX.exists():
        subprocess.run(["python", "scripts/export.py",
                        "--weights", str(PF_SAFE), "--out", str(PF_ONNX)], cwd=ROOT, check=True)
    import cv2

    from data.preprocess import preprocess
    from model.weights import apply_meta, load_meta, load_weights

    m = _build(True)
    apply_meta(m, load_meta(PF_SAFE))
    load_weights(m, PF_SAFE, strict=True)
    m.eval()
    m.model[-1].end2end = True
    m.model[-1].export = True  # 与导出口径一致：conf=0 静态路径

    img = cv2.imread(str(TEST_IMAGE))
    tensor, _, _ = preprocess(img)
    with torch.no_grad():
        r, _ = m(tensor)
    sess = ort.InferenceSession(str(PF_ONNX), providers=["CPUExecutionProvider"])
    o, _ = sess.run(None, {"images": tensor.numpy()})
    r, o = r.numpy()[0], np.asarray(o)[0]
    r, o = r[r[:, 4] > 0.25], o[o[:, 4] > 0.25]
    assert len(r) == len(o), f"有效检测数不一致: pt {len(r)} vs onnx {len(o)}"
    order = np.lexsort((r[:, 0], r[:, 1], -r[:, 4]))
    r, o = r[order], o[order]
    assert np.abs(r[:, :4] - o[:, :4]).max() < 1e-3, "boxes 超阈值"
    assert np.abs(r[:, 4] - o[:, 4]).max() < 1e-3, "scores 超阈值"
    assert np.abs(r[:, 6:] - o[:, 6:]).max() < 1e-3, "mask coefficients 超阈值"
    assert (r[:, 5] == o[:, 5]).all(), "conf>0.25 出现类别翻转"
