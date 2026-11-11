"""Flash-YOLO-pose 结构验收：参数数 / 键面 / 前向契约 / 命名与任务分派

flash-yolo-pose = flash-yolo 的 backbone/neck + Pose26 头；不依赖官方权重，纯结构测试。
"""

import numpy as np
import torch

from config.train_config import TrainConfig
from model.build import YOLO26_FAMILY, arch_display_name, arch_task, build_model
from model.head import Pose26
from model.weights import arch_from_weights, arch_has_scale, resolve_arch_scale, save_weights
from train.loss import ComputeLossPose, build_loss

BODY_LAYERS = 24  # 0..23 为 backbone/neck；24 为任务头


def test_structure_and_params():
    """结构硬指标：3,081,432 参数（Pose26 头 1,416,840，含训练期 sigma/flow）"""
    model = build_model("flash-yolo-pose")
    head = model.model[-1]
    assert isinstance(head, Pose26)
    assert head.kpt_shape == (17, 3)
    assert sum(p.numel() for p in model.parameters()) == 3_081_432
    assert sum(p.numel() for p in head.parameters()) == 1_416_840


def test_backbone_neck_identical_to_detect():
    """backbone+neck 与 flash-yolo 逐键同构——姿态版只换最后一行头"""
    torch.manual_seed(0)
    det = build_model("flash-yolo").state_dict()
    torch.manual_seed(0)
    pose = build_model("flash-yolo-pose").state_dict()
    det_body = {k for k in det if not k.startswith(f"model.{BODY_LAYERS}.")}
    pose_body = {k for k in pose if not k.startswith(f"model.{BODY_LAYERS}.")}
    assert det_body == pose_body
    assert all(torch.equal(det[k], pose[k]) for k in det_body)


def test_forward_contract_e2e():
    """E2E：关键点与框共享 top-k，输出 (B, 300, 6+nk)"""
    model = build_model("flash-yolo-pose").eval()
    model.model[-1].end2end = True
    with torch.no_grad():
        out = model(torch.rand(1, 3, 640, 640))
    assert out.shape == (1, 300, 57)


def test_forward_contract_raw():
    """raw（end2end=False）：解码后的 xywh + 分数 + 关键点，(B, 4+nc+nk, N)"""
    model = build_model("flash-yolo-pose").eval()
    model.model[-1].end2end = False
    with torch.no_grad():
        raw = model(torch.rand(1, 3, 640, 640))
    assert raw.shape == (1, 56, 8400)


def test_custom_nc():
    """自定义类别数：nc 覆盖 yaml 结构值，关键点分支不变"""
    model = build_model("flash-yolo-pose", nc=3)
    head = model.model[-1]
    assert head.nc == 3
    assert head.cv3[0][2].out_channels == 3
    assert head.cv4_kpts[0].out_channels == 51


def test_arch_resolution_and_task():
    """命名/任务轴：固定档 flash，文件名可反推，且不误吸 flash-yolo / flash-yolo-seg"""
    assert arch_from_weights("weights/flash-yolo-pose.safetensors") == "flash-yolo-pose"
    assert arch_from_weights("flash-yolo-pose.pt") == "flash-yolo-pose"
    assert resolve_arch_scale("weights/flash-yolo-pose.safetensors") == ("flash-yolo-pose", None)
    assert arch_has_scale("flash-yolo-pose") is False
    assert arch_task("flash-yolo-pose") == "pose"
    assert arch_display_name("flash-yolo-pose", "flash") == "Flash-YOLO-pose"
    assert "flash-yolo-pose" in YOLO26_FAMILY
    assert arch_from_weights("weights/flash-yolo.safetensors") == "flash-yolo"
    assert arch_from_weights("weights/flash-yolo-seg.safetensors") == "flash-yolo-seg"


def test_loss_dispatch():
    """损失分派：flash-yolo-pose 复用 ComputeLossPose"""
    model = build_model("flash-yolo-pose")
    loss = build_loss("flash-yolo-pose", TrainConfig(), model.model[-1], torch.device("cpu"))
    assert isinstance(loss, ComputeLossPose)


def test_engine_reads_pose_metadata(tmp_path):
    """权重 metadata 携带 kpt_shape；PtEngine 按 pose 任务建图并 strict load"""
    from utils.engines.pt import PtEngine

    model = build_model("flash-yolo-pose")
    path = tmp_path / "flash-yolo-pose.safetensors"
    save_weights(model, path, metadata={
        "arch": "flash-yolo-pose", "scale": "flash", "nc": 1,
        "names": ["person"], "kpt_shape": [17, 3], "imgsz": 160,
    })
    engine = PtEngine(str(path), model="flash-yolo-pose", scale=None, imgsz=160)
    assert engine.task == "pose"
    assert engine.kpt_shape == (17, 3)
    assert engine.model.model[-1].kpt_shape == (17, 3)
    assert engine.predict(np.zeros((160, 160, 3), np.uint8)).keypoints is not None
