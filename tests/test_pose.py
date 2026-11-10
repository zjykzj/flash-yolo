"""YOLO26-pose：结构 / strict load / 与官方前向逐位对齐 / 导出分支回归

需要 dev-time 权重（缺失时自动 skip）：
    python scripts/download_weights.py --model yolo26n-pose
    python scripts/convert_weights.py --src weights/yolo26n-pose.pt \
        --dst weights/yolo26n-pose.safetensors
"""

import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
PT_PATH = ROOT / "weights" / "yolo26n-pose.pt"
SAFE_PATH = ROOT / "weights" / "yolo26n-pose.safetensors"
RAW_ONNX_PATH = ROOT / "weights" / "yolo26n-pose_raw.onnx"
EXPECTED_PARAMS = 3_679_464  # 官方 yolo26n-pose（含 flow_model 参数）


def _build_ours(end2end=True):
    from model.build import DetectionModel, YOLO26_POSE_CONFIG_PATH
    from model.weights import load_weights

    model = DetectionModel(YOLO26_POSE_CONFIG_PATH, scale="n").eval()
    load_weights(model, SAFE_PATH, strict=True)
    model.model[-1].end2end = end2end
    return model


def test_param_count():
    """结构硬指标：yolo26n-pose = 3,679,464 参数（与官方 checkpoint 一致）"""
    from model.build import DetectionModel, YOLO26_POSE_CONFIG_PATH

    model = DetectionModel(YOLO26_POSE_CONFIG_PATH, scale="n")
    assert sum(p.numel() for p in model.parameters()) == EXPECTED_PARAMS


def test_arch_and_metadata():
    """文件名识别 + 转换产物 metadata：kpt_shape / nc 必须随权重走"""
    from model.weights import arch_from_weights, load_meta, scale_from_weights

    assert arch_from_weights("yolo26n-pose.safetensors") == "yolo26-pose"
    assert arch_from_weights("yolo26s-pose.pt") == "yolo26-pose"
    assert scale_from_weights("yolo26n-pose.safetensors") == "n"
    if SAFE_PATH.exists():
        meta = load_meta(SAFE_PATH)
        assert meta["arch"] == "yolo26-pose"
        assert meta["kpt_shape"] == [17, 3]
        assert meta["nc"] == 1


def test_strict_load():
    """strict load 全 key 匹配（含 cv4_kpts / cv4_sigma / flow_model）"""
    if not SAFE_PATH.exists():
        pytest.skip("需要先转换 yolo26n-pose.pt -> safetensors")
    from model.build import DetectionModel, YOLO26_POSE_CONFIG_PATH
    from model.weights import load_weights

    model = DetectionModel(YOLO26_POSE_CONFIG_PATH, scale="n")
    missing, unexpected = load_weights(model, SAFE_PATH, strict=False)
    assert not missing, f"missing keys: {missing[:5]}"
    assert not unexpected, f"unexpected keys: {unexpected[:5]}"


def test_numeric_alignment():
    """同输入下与官方 Pose26 逐位一致：E2E (1,300,57) 与 raw (1,56,8400)"""
    if not (SAFE_PATH.exists() and PT_PATH.exists()):
        pytest.skip("需要官方 .pt 与转换后的 .safetensors")
    ultralytics = pytest.importorskip("ultralytics")

    torch.manual_seed(0)
    x = torch.randn(1, 3, 640, 640)
    ours = _build_ours(end2end=True)
    official = ultralytics.YOLO(str(PT_PATH)).model.eval()

    with torch.no_grad():
        official.model[-1].end2end = True
        ref = official(x)
        ref = ref[0] if isinstance(ref, (tuple, list)) else ref
        out = ours(x)
        diff = (ref - out).abs().max().item()
        assert diff < 1e-4, f"E2E 输出误差 {diff:.2e} 超阈值"

        official.model[-1].end2end = False
        ref_raw = official(x)
        ref_raw = ref_raw[0] if isinstance(ref_raw, (tuple, list)) else ref_raw
        ours.model[-1].end2end = False
        out_raw = ours(x)
        diff_raw = (ref_raw - out_raw).abs().max().item()
        assert diff_raw < 1e-4, f"raw 输出误差 {diff_raw:.2e} 超阈值"
    print(f"  E2E 误差 {diff:.2e}；raw 误差 {diff_raw:.2e}")


def test_export_kpts_decode_branch_is_equivalent():
    """ONNX export 分支（view 表达）与普通 in-place 解码逐位一致

    回归：曾把 (2,N) anchors 先 .T 再 reshape，转置视图 reshape 会按物理序重排，
    ONNX 关键点坐标整体翻倍；该测试不依赖 onnxruntime，直接跑 PyTorch 两分支。
    """
    if not SAFE_PATH.exists():
        pytest.skip("需要转换后的权重")
    model = _build_ours(end2end=True)
    head = model.model[-1]
    torch.manual_seed(0)
    x = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        ref = model(x)
        head.export = True
        out = model(x)
        head.export = False
    assert torch.equal(ref, out), f"export 分支与普通解码不一致: {(ref - out).abs().max().item():.2e}"


def test_route_helpers():
    """任务轴/家族门控：pose 归 yolo26 家族，task=pose"""
    from model.build import YOLO26_FAMILY, arch_task

    assert arch_task("yolo26-pose") == "pose"
    assert "yolo26-pose" in YOLO26_FAMILY


def test_label_lines_and_draw():
    """推理侧 pose 标签 = cls + 框 + 17×(x,y,v)；可视化画关键点/骨架不崩"""
    from scripts.infer import _label_lines
    from utils.engines import Detections
    from utils.visualize import draw_detections

    kpts = np.zeros((1, 17, 3), np.float32)
    kpts[0, :, 2] = 0.9
    dets = Detections(boxes=np.array([[10, 20, 50, 80]], np.float32),
                      scores=np.array([0.9], np.float32),
                      class_ids=np.array([0], np.int64), keypoints=kpts)
    lines = _label_lines(dets, "pose", 100, 200)
    assert len(lines) == 1
    assert len(lines[0].split()) == 5 + 17 * 3
    out = draw_detections(np.zeros((100, 100, 3), np.uint8), dets, names={0: "person"})
    assert out.any()


def test_pose_export_parity_raw():
    """raw pose ONNX 与 pt 一致（解码后的 xywh/scores/keypoints）

    与检测 raw 的 ltrb 不同，pose raw 输出的是**解码到像素**的坐标（~0..1000），
    ONNX 与 PyTorch 的累加顺序差会被 stride/anchors 放大到 ~1e-2 px；相对误差 ~1e-6，
    对检测框/关键点无意义。阈值按绝对像素 1e-2（均值实测 ~3e-5）。
    """
    if not SAFE_PATH.exists():
        pytest.skip("需要转换后的权重")
    image_path = ROOT / "assets" / "traffic.jpg"
    if not image_path.exists():
        pytest.skip("需要 assets/traffic.jpg")
    if not RAW_ONNX_PATH.exists():
        subprocess.run(["python", "scripts/export.py", "--weights", str(SAFE_PATH),
                        "--out", str(RAW_ONNX_PATH), "--raw"], cwd=ROOT, check=True)
    ort = pytest.importorskip("onnxruntime")
    sess = ort.InferenceSession(str(RAW_ONNX_PATH), providers=["CPUExecutionProvider"])

    import cv2

    from data.preprocess import preprocess
    from model.build import DetectionModel, YOLO26_POSE_CONFIG_PATH
    from model.weights import load_weights

    model = DetectionModel(YOLO26_POSE_CONFIG_PATH, scale="n").eval()
    load_weights(model, SAFE_PATH, strict=True)
    model.model[-1].end2end = False
    image = cv2.imread(str(image_path))
    x, _, _ = preprocess(image, 640)
    with torch.no_grad():
        ref = model(x).numpy()
    out = sess.run(None, {"images": x.numpy()})[0]
    diff = float(np.abs(ref - out).max())
    assert diff < 1e-2, f"pose raw: onnx vs pt 误差 {diff:.2e} 超阈值"
    print(f"  pose raw onnx vs pt 误差: {diff:.2e}")
