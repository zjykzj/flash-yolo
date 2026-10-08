"""YOLOv3-tiny 结构验收：层数/参数/前向形状/解码数值/路由与缓冲"""

import math

import pytest
import torch

from model.build import build_model, build_yolov3_tiny
from model.head_v3 import decode_level
from model.weights import load_weights, save_weights


def _build(nc=80, imgsz=640):
    return build_model("yolov3-tiny", imgsz=imgsz, nc=nc)


def test_structure_and_params():
    """21 层、8,852,366 参数（nc=80）、anchors/stride 为非持续 buffer、路由归一化正确"""
    model = _build()
    assert len(model.model) == 21
    assert sum(p.numel() for p in model.parameters()) == 8_852_366
    # 非持续 buffer：不进 state_dict（转换器/EMA/strict load 零特判）
    sd_keys = set(model.state_dict())
    assert not any("anchors" in k or "stride" in k for k in sd_keys)
    assert len(sd_keys) == 11 * 6 + 4  # 11 颗 LeakyConv（conv + BN 五项）+ 头两颗 1×1（weight+bias）
    # 路由：层16 的 -2 归一化为绝对 14；被引用的层全部登记 save（含 Concat 的 17）
    assert model.model[16].f == 14
    assert model.model[20].f == [19, 15]
    assert model.save == {8, 14, 15, 17, 19}
    print("  结构正确：21 层 / 8,852,366 参数 / save={8,14,15,17,19}")


def test_forward_shapes():
    """train 口径两级 raw；eval 口径解码 (B, NA, 5+nc)（640→6000、416→2535）"""
    model = _build().train()
    with torch.no_grad():
        outs = model(torch.randn(2, 3, 640, 640))
    assert [tuple(o.shape) for o in outs] == [(2, 255, 40, 40), (2, 255, 20, 20)]  # 级序 [P4/16, P5/32]

    model.eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, 640, 640))
    assert out.shape == (2, 6000, 85)
    with torch.no_grad():
        out416 = _build(imgsz=416).eval()(torch.randn(1, 3, 416, 416))
    assert out416.shape == (1, 2535, 85)
    print("  前向形状正确：raw (2,255,20,20)/(2,255,40,40)，解码 (2,6000,85) / (1,2535,85)")


def test_forward_feats():
    """backbone+neck 前向返回 head 输入列表 [P4(256ch,40×40), P5(512ch,20×20)]"""
    model = _build().eval()
    with torch.no_grad():
        feats = model.forward_feats(torch.randn(2, 3, 640, 640))
    assert [tuple(f.shape) for f in feats] == [(2, 256, 40, 40), (2, 512, 20, 20)]
    print("  forward_feats 形状正确")


def test_decode_values():
    """解码数值（darknet 口径手算）：σ 中心 + 格 + exp 尺度 × 锚

    构造 1×1 网格的合成 raw（imgsz=416 → 锚不缩放：P4 首槽 (23,27)、stride 16）：
    tx=ty=0 → 中心 (0.5·16, 0.5·16) = (8,8)；tw=ln2 → w = 2×23 = 46；th=0 → h = 27。
    """
    raw = torch.zeros(1, 255, 1, 1)
    raw[0, 2, 0, 0] = math.log(2.0)  # 锚 0 的 tw（通道 = a*85 + 2）
    raw[0, 4, 0, 0] = 0.0            # obj logit
    raw[0, 5, 0, 0] = 100.0          # 类别 0 logit
    anchors = torch.tensor([[23.0, 27.0], [37.0, 58.0], [81.0, 82.0]])
    boxes, obj, cls = decode_level(raw, anchors, torch.tensor(16.0), no=85)
    x1, y1, x2, y2 = boxes[0, 0, 0, 0].tolist()
    assert abs(x1 - (8 - 23)) < 1e-5 and abs(x2 - (8 + 23)) < 1e-5, (x1, x2)
    assert abs(y1 - (8 - 13.5)) < 1e-5 and abs(y2 - (8 + 13.5)) < 1e-5, (y1, y2)
    assert abs(obj[0, 0, 0, 0].sigmoid().item() - 0.5) < 1e-6
    assert abs(cls[0, 0, 0, 0, 0].sigmoid().item() - 1.0) < 1e-6
    # 未设置的槽位：中心 (8,8)、宽高 = 锚原值（exp(0)=1）
    x1b, y1b, x2b, y2b = boxes[0, 1, 0, 0].tolist()
    assert abs(x2b - x1b - 37.0) < 1e-5 and abs(y2b - y1b - 58.0) < 1e-5, (x1b, y1b, x2b, y2b)
    print("  解码数值正确（σ/格/exp/锚）")


def test_anchors_imgsz_invariant():
    """anchors 照官方原值、不随 imgsz 缩放（官方权重多尺度训练由 tw 自补偿输入尺度；
    线性缩放到 640 实测把 500 图 mAP50 从 0.41 打到 0.15）"""
    head416 = _build(imgsz=416).model[-1]
    head640 = _build(imgsz=640).model[-1]
    assert torch.equal(head416.anchors, head640.anchors)
    assert abs(head640.anchors[0, 0, 0].item() - 23.0) < 1e-6
    print("  anchors 原值保留（不随 imgsz 缩放）")


def test_unknown_scale_rejected():
    """v3-tiny 只有 tiny 档：传其他 scale 报明确错误"""
    with pytest.raises(ValueError, match="scale 'n' not in \\['tiny'\\]"):
        build_model("yolov3-tiny", scale="n")
    with pytest.raises(ValueError, match="unknown arch"):
        build_model("yolo99")
    print("  档位/架构名校验正确")


def test_strict_roundtrip(tmp_path):
    """safetensors 往返：严格加载键集一致、参数逐位相同"""
    model = build_yolov3_tiny()
    path = tmp_path / "v3.safetensors"
    save_weights(model, path)
    fresh = build_yolov3_tiny()
    missing, unexpected = load_weights(fresh, path, strict=True)
    assert not missing and not unexpected
    for (n1, p1), (n2, p2) in zip(model.state_dict().items(), fresh.state_dict().items()):
        assert n1 == n2 and torch.equal(p1, p2), f"参数不一致: {n1}"
    print(f"  safetensors 往返一致（{len(model.state_dict())} 个张量）")
