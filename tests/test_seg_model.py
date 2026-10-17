"""YOLO26-seg 结构验收：参数数 / 键面 / 前向契约 / 与官方权重位级对拍

需要权重时先准备（dev-time，缺失自动跳过）：
    python scripts/convert_weights.py --src weights/yolo26n-seg.pt \
        --dst weights/yolo26n-seg.safetensors
"""

from pathlib import Path

import pytest
import torch

from model.build import build_model, build_yolo26, build_yolo26_seg, arch_task
from model.head import Proto26, Segment26
from model.weights import arch_from_weights, load_weights, resolve_arch_scale

ROOT = Path(__file__).resolve().parent.parent
PT_PATH = ROOT / "weights" / "yolo26n-seg.pt"
SAFE_PATH = ROOT / "weights" / "yolo26n-seg.safetensors"


def test_structure_and_params():
    """结构硬指标：yolo26n-seg = 3,126,280 参数（含训练期 semseg 辅助头，推理不使用）"""
    model = build_yolo26_seg("n")
    head = model.model[-1]
    assert isinstance(head, Segment26) and isinstance(head.proto, Proto26)
    assert (head.nm, head.npr) == (32, 64)  # npr 随 width 缩放（yaml 写 256）
    assert sum(p.numel() for p in model.parameters()) == 3_126_280
    assert sum(p.numel() for p in head.proto.semseg.parameters()) == 79_184


def test_backbone_neck_identical_to_detect():
    """backbone+neck 与检测模型逐键同构——分割只改头（官方 yaml 的唯一差异也是这一行）

    同 seed 分别构建两个模型：backbone/neck 的层序与初始化顺序一致，应逐位相同；
    头部之后才分叉，因此差异必然只可能出现在 model.23.*。
    """
    torch.manual_seed(0)
    det = build_yolo26("n").state_dict()
    torch.manual_seed(0)
    seg = build_yolo26_seg("n").state_dict()
    det_body = {k for k in det if not k.startswith("model.23.")}
    seg_body = {k for k in seg if not k.startswith("model.23.")}
    assert det_body == seg_body
    assert all(torch.equal(det[k], seg[k]) for k in det_body)


def test_forward_contract_e2e():
    """E2E：((B, max_det, 6+nm), proto)，系数与原型形状按 640 输入核验"""
    model = build_yolo26_seg("n")
    with torch.no_grad():
        dets, proto = model(torch.rand(1, 3, 640, 640))
    assert dets.shape == (1, 300, 38)  # [xyxy, conf, cls, 32 系数]
    assert proto.shape == (1, 32, 160, 160)  # 1/4 输入分辨率


def test_forward_contract_raw():
    """raw（end2end=False）：((B, 4+nc+nm, N), proto)"""
    model = build_yolo26_seg("n")
    model.model[-1].end2end = False
    with torch.no_grad():
        raw, proto = model(torch.rand(1, 3, 640, 640))
    assert raw.shape == (1, 4 + 80 + 32, 8400)
    assert proto.shape == (1, 32, 160, 160)


def test_e2e_selection_picks_matching_coefficients():
    """E2E 的系数必须取自同一次 top-k：与手动按锚点索引重算逐位一致"""
    model = build_yolo26_seg("n")
    head = model.model[-1]
    x = [torch.rand(1, c, 640 // s, 640 // s) for c, s in zip((64, 128, 256), (8, 16, 32))]
    with torch.no_grad():
        y = head._forward_head_seg(x, head.one2one_cv2, head.one2one_cv3, head.one2one_cv4)
        out = head._e2e_postprocess(y, x)
    assert out.shape == (1, 300, 38)
    # 系数（末 32 列）必须能在原始系数张量里按行逐位找到（随机权重下匹配几乎唯一；
    # 取前 4 列做键以控制 (300, N, 4) 的中间张量大小）
    coeff = y["mask_coefficient"].permute(0, 2, 1)[0]  # (N, 32)
    hits = (coeff.unsqueeze(0)[..., :4] == out[0, :, 6:10].unsqueeze(1)).all(-1).any(1)
    assert bool(hits.all())


def test_train_branch_contract():
    """训练模式：双分支 dict，o2o 的 proto 走 detach（梯度只经 o2m 流回原型）"""
    model = build_yolo26_seg("n")
    model.train()
    out = model(torch.rand(1, 3, 640, 640))
    assert set(out) == {"one2many", "one2one"}
    for branch in out.values():
        assert set(branch) == {"boxes", "scores", "mask_coefficient", "proto", "semantic"}
        assert branch["mask_coefficient"].shape == (1, 32, 8400)
        assert branch["semantic"].shape == (1, 80, 80, 80)  # P3/8 的语义辅助头
    assert out["one2many"]["proto"].requires_grad
    assert not out["one2one"]["proto"].requires_grad
    assert out["one2many"]["semantic"].requires_grad
    assert not out["one2one"]["semantic"].requires_grad


def test_arch_resolution():
    """架构轴：文件名/后缀 -> arch + task（档位正则仍从 yolo26n-seg 里取 n）"""
    assert arch_from_weights("weights/yolo26n-seg.safetensors") == "yolo26-seg"
    assert arch_from_weights("yolo26x-seg.pt") == "yolo26-seg"
    assert arch_from_weights("weights/yolo26n.safetensors") is None  # 检测权重仍走默认 yolo26
    assert resolve_arch_scale("weights/yolo26n-seg.safetensors") == ("yolo26-seg", "n")
    assert arch_task("yolo26-seg") == "segment"
    assert arch_task("yolo26") == "detect"


def test_custom_nc():
    """自定义类别数：nc 覆盖 yaml 结构值，头与原型同口径"""
    model = build_yolo26_seg("n", nc=3)
    head = model.model[-1]
    assert head.nc == 3
    assert head.cv3[0][2].out_channels == 3
    assert head.proto.semseg[2].out_channels == 3


def test_strict_load_official_weights():
    """官方转换产物 strict load（成功即键面逐键一致）"""
    if not SAFE_PATH.exists():
        pytest.skip("需要 weights/yolo26n-seg.safetensors（见文件头注释）")
    model = build_yolo26_seg("n")
    missing, unexpected = load_weights(model, SAFE_PATH, strict=False)
    assert not missing and not unexpected


def test_numeric_alignment_with_official():
    """同权重同输入：E2E 检测输出与 proto 与官方逐位一致（实测 max|Δ| = 0）"""
    if not (SAFE_PATH.exists() and PT_PATH.exists()):
        pytest.skip("需要官方 .pt 与转换后的 .safetensors")
    pytest.importorskip("ultralytics")

    from ultralytics.nn.tasks import SegmentationModel

    official = SegmentationModel("yolo26n-seg.yaml", ch=3, nc=80, verbose=False)
    ckpt = torch.load(PT_PATH, map_location="cpu", weights_only=False)
    sd = ckpt["model"].float().state_dict()
    official.load_state_dict(sd, strict=True)
    official.eval()
    official.model[-1].export = True
    official.model[-1].end2end = True

    ours = build_yolo26_seg("n")
    ours.load_state_dict(sd, strict=True)

    torch.manual_seed(0)
    x = torch.rand(1, 3, 640, 640)
    with torch.no_grad():
        dets_o, proto_o = official(x)
        dets_m, proto_m = ours(x)
    assert (dets_o - dets_m).abs().max().item() < 1e-6
    assert (proto_o - proto_m).abs().max().item() < 1e-6


def test_unknown_scale_rejected():
    with pytest.raises(ValueError, match="scale"):
        build_model("yolo26-seg", scale="tiny")
