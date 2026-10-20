"""Flash-YOLO-seg 结构验收：参数数 / 键面 / 前向契约 / 命名与任务分派

flash-yolo-seg = flash-yolo 的 backbone/neck + Segment26 头；不依赖官方权重，纯结构测试。
"""

import torch

from config.train_config import TrainConfig
from model.build import arch_display_name, arch_task, build_model
from model.head import Proto26, Segment26
from model.weights import arch_from_weights, arch_has_scale, resolve_arch_scale
from train.loss import ComputeLossSeg, build_loss

BODY_LAYERS = 24  # 0..23 为 backbone/neck；24 为任务头


def test_structure_and_params():
    """结构硬指标：2,528,248 参数（含训练期 semseg 辅助头）；nm=32 / npr=64（= yolo26n 生效值）"""
    model = build_model("flash-yolo-seg")
    head = model.model[-1]
    assert isinstance(head, Segment26) and isinstance(head.proto, Proto26)
    assert (head.nm, head.npr) == (32, 64)  # flash 档 width=1.0，yaml 直接写生效值 64
    assert sum(p.numel() for p in model.parameters()) == 2_528_248
    assert sum(p.numel() for p in head.parameters()) == 863_656
    assert sum(p.numel() for p in head.proto.semseg.parameters()) == 79_184


def test_backbone_neck_identical_to_detect():
    """backbone+neck 与 flash-yolo 逐键同构——分割只换最后一行头"""
    torch.manual_seed(0)
    det = build_model("flash-yolo").state_dict()
    torch.manual_seed(0)
    seg = build_model("flash-yolo-seg").state_dict()
    det_body = {k for k in det if not k.startswith(f"model.{BODY_LAYERS}.")}
    seg_body = {k for k in seg if not k.startswith(f"model.{BODY_LAYERS}.")}
    assert det_body == seg_body
    assert all(torch.equal(det[k], seg[k]) for k in det_body)


def test_forward_contract_e2e():
    """E2E：((B, max_det, 6+nm), proto)，640 输入下形状与 yolo26-seg 同契约"""
    model = build_model("flash-yolo-seg").eval()
    model.model[-1].end2end = True
    with torch.no_grad():
        dets, proto = model(torch.rand(1, 3, 640, 640))
    assert dets.shape == (1, 300, 38)
    assert proto.shape == (1, 32, 160, 160)


def test_forward_contract_raw():
    """raw（end2end=False）：((B, 4+nc+nm, N), proto)"""
    model = build_model("flash-yolo-seg").eval()
    model.model[-1].end2end = False
    with torch.no_grad():
        raw, proto = model(torch.rand(1, 3, 640, 640))
    assert raw.shape == (1, 4 + 80 + 32, 8400)
    assert proto.shape == (1, 32, 160, 160)


def test_train_branch_contract():
    """训练模式：双分支 dict，o2o 的 proto/semantic 均 detach（梯度只经 o2m 回流）"""
    model = build_model("flash-yolo-seg")
    model.train()
    out = model(torch.rand(1, 3, 640, 640))
    assert set(out) == {"one2many", "one2one"}
    for branch in out.values():
        assert set(branch) == {"boxes", "scores", "mask_coefficient", "proto", "semantic"}
        assert branch["mask_coefficient"].shape == (1, 32, 8400)
        assert branch["semantic"].shape == (1, 80, 80, 80)
    assert out["one2many"]["proto"].requires_grad
    assert not out["one2one"]["proto"].requires_grad
    assert out["one2many"]["semantic"].requires_grad
    assert not out["one2one"]["semantic"].requires_grad


def test_custom_nc():
    """自定义类别数：nc 覆盖 yaml 结构值，头与原型同口径"""
    model = build_model("flash-yolo-seg", nc=3)
    head = model.model[-1]
    assert head.nc == 3
    assert head.cv3[0][2].out_channels == 3
    assert head.proto.semseg[2].out_channels == 3


def test_arch_resolution_and_task():
    """命名/任务轴：flash-yolo-seg 固定档 flash，文件名可反推，且不误吸 flash-yolo"""
    assert arch_from_weights("weights/flash-yolo-seg.safetensors") == "flash-yolo-seg"
    assert arch_from_weights("flash-yolo-seg.pt") == "flash-yolo-seg"
    assert resolve_arch_scale("weights/flash-yolo-seg.safetensors") == ("flash-yolo-seg", None)
    assert arch_has_scale("flash-yolo-seg") is False
    assert arch_task("flash-yolo-seg") == "segment"
    assert arch_display_name("flash-yolo-seg", "flash") == "Flash-YOLO-seg"
    # 旧检测名不被新后缀误吸
    assert arch_from_weights("weights/flash-yolo.safetensors") == "flash-yolo"
    assert resolve_arch_scale("weights/flash-yolo.safetensors") == ("flash-yolo", None)


def test_loss_dispatch():
    """损失分派：flash-yolo-seg 复用 ComputeLossSeg"""
    model = build_model("flash-yolo-seg")
    loss = build_loss("flash-yolo-seg", TrainConfig(), model.model[-1], torch.device("cpu"))
    assert isinstance(loss, ComputeLossSeg)
