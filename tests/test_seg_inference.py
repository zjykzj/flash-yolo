"""YOLO26-seg 推理链验收：NMS 系数透传 / 掩码解码与官方逐像素一致 / 三引擎契约

需要权重与图片的用例在缺失时自动跳过（准备见 tests/test_seg_model.py 头部注释）。
"""

from pathlib import Path

import numpy as np
import pytest

from utils.postprocess import masks_to_segments, nms_per_image, process_mask

ROOT = Path(__file__).resolve().parent.parent
SAFE_PATH = ROOT / "weights" / "yolo26n-seg.safetensors"
PT_PATH = ROOT / "weights" / "yolo26n-seg.pt"
IMG_PATH = ROOT / "assets" / "traffic.jpg"


def _letterbox_prep(imgsz=640):
    import cv2

    from data.preprocess import preprocess

    img = cv2.imread(str(IMG_PATH))
    x, ratio, pad = preprocess(img, imgsz)
    return img, x, ratio, pad


# ---- 纯 numpy 口径（不需权重）----


def test_masks_to_segments_contours_and_merge():
    """单连通 -> 一条外轮廓；多连通 -> 合并成一条；空掩码 -> 空数组"""
    masks = np.zeros((3, 40, 60), np.uint8)
    masks[0, 5:20, 5:25] = 1
    masks[1, 4:12, 4:15] = 1
    masks[1, 25:35, 40:55] = 1
    segs = masks_to_segments(masks)
    assert len(segs) == 3
    assert segs[0].shape[1] == 2 and len(segs[0]) >= 4
    x0, y0 = segs[0].min(0)
    x1, y1 = segs[0].max(0)
    assert (x0, y0, x1, y1) == (5, 5, 24, 19)  # 像素中心轮廓
    # 两段合并后的多边形跨越两个连通域
    assert segs[1][:, 0].max() >= 50 and segs[1][:, 1].min() <= 5
    assert len(segs[1]) >= 8
    assert segs[2].shape == (0, 2)


def test_label_lines_seg_polygon():
    """分割推理标签写归一化多边形（少于 3 点跳过），检测路径保持框格式"""
    from scripts.infer import _label_lines
    from utils.engines import Detections

    masks = np.zeros((2, 20, 20), np.uint8)
    masks[0, 2:10, 4:12] = 1
    dets = Detections(np.array([[4, 2, 12, 10], [0, 0, 5, 5]], np.float32),
                      np.array([0.9, 0.8], np.float32), np.array([3, 7]), masks)
    lines = _label_lines(dets, "segment", 20, 20)
    assert len(lines) == 1  # 空掩码实例不写行
    parts = lines[0].split()
    assert int(parts[0]) == 3 and len(parts) >= 7 and (len(parts) - 1) % 2 == 0
    coords = np.array([float(v) for v in parts[1:]])
    assert coords.min() >= 0 and coords.max() <= 1
    box_lines = _label_lines(dets, "detect", 20, 20)
    assert len(box_lines) == 2 and len(box_lines[0].split()) == 5


def test_nms_per_image_carries_extra():
    """extra 行必须与保留框一一对应（分割的 mask 系数随同一次 NMS 选择取回）"""
    rng = np.random.default_rng(0)
    boxes = rng.uniform(0, 100, size=(20, 4))
    boxes[:, 2:] = boxes[:, :2] + 30  # 保证 xyxy 合法
    score_map = np.zeros((20, 3), dtype=np.float32)
    score_map[rng.integers(0, 20, 8), rng.integers(0, 3, 8)] = 0.9
    extra = np.arange(20 * 5, dtype=np.float32).reshape(20, 5)

    det, extra_kept = nms_per_image(boxes, score_map, conf_thres=0.5, max_det=100, extra=extra)
    assert len(det) == len(extra_kept)
    # 每个保留框的系数行，应与"在（分数、类、坐标）上匹配的候选行"一致
    for row, e in zip(det, extra_kept):
        cand = np.nonzero((np.abs(extra - e).max(axis=1) == 0))[0]
        assert len(cand) == 1, "系数行必须唯一对应一个候选"
        hit = cand[0]
        assert np.allclose(boxes[hit], row[:4]), "系数行与原候选框必须同源"


def test_nms_per_image_extra_empty_paths():
    """无候选 / max_det<=0 时 extra 的形状正确（空栈不崩）"""
    boxes = np.zeros((0, 4), dtype=np.float32)
    scores = np.zeros((0, 2), dtype=np.float32)
    extra = np.zeros((0, 7), dtype=np.float32)
    det, extra_kept = nms_per_image(boxes, scores, extra=extra)
    assert det.shape == (0, 6) and extra_kept.shape == (0, 7)

    boxes = np.array([[0, 0, 10, 10]], dtype=np.float32)
    scores = np.array([[0.9, 0.0]], dtype=np.float32)
    extra = np.ones((1, 4), dtype=np.float32)
    det, extra_kept = nms_per_image(boxes, scores, max_det=0, extra=extra)
    assert det.shape == (0, 6) and extra_kept.shape == (0, 4)


def test_process_mask_empty_detections():
    """零检出：返回 (0, H, W) uint8 空栈（F.interpolate 不接受 N=0，必须短路）"""
    proto = np.zeros((32, 160, 160), dtype=np.float32)
    coeffs = np.zeros((0, 32), dtype=np.float32)
    boxes = np.zeros((0, 4), dtype=np.float32)
    masks = process_mask(proto, coeffs, boxes, (640, 640), (427, 640), 1.0, (0, 0))
    assert masks.shape == (0, 427, 640) and masks.dtype == np.uint8


def test_process_mask_shape_and_crop():
    """掩码形状 = 原图；且被裁到框内（框外必为 0）"""
    rng = np.random.default_rng(0)
    proto = rng.standard_normal((32, 160, 160)).astype(np.float32) * 4  # 让 logits 有正有负
    coeffs = rng.standard_normal((3, 32)).astype(np.float32)
    boxes = np.array([[10, 20, 100, 90], [200, 30, 300, 200], [5, 5, 50, 60]], dtype=np.float32)
    masks = process_mask(proto, coeffs, boxes, (640, 640), (120, 260), 1.0, (0, 0))
    assert masks.shape == (3, 120, 260)
    for m, (x1, y1, x2, y2) in zip(masks, boxes):
        ys, xs = np.nonzero(m)
        if len(xs) == 0:
            continue
        assert xs.min() >= int(x1) and xs.max() < int(np.ceil(x2))
        assert ys.min() >= int(y1) and ys.max() < int(np.ceil(y2))


# ---- 需要官方权重的用例 ----


@pytest.mark.skipif(not SAFE_PATH.exists(), reason="需要 weights/yolo26n-seg.safetensors")
def test_pt_engine_seg_contract():
    """pt 引擎：Detections.masks 为 (N, H, W) uint8 @原图分辨率，行数与框一致"""
    import cv2

    from utils.engines import PtEngine

    img = cv2.imread(str(IMG_PATH))
    engine = PtEngine(str(SAFE_PATH), scale="n", model="yolo26-seg", device="cpu")
    dets = engine.predict(img, conf_thres=0.25)
    assert len(dets.boxes) > 0
    assert dets.masks is not None
    assert dets.masks.shape == (len(dets.boxes), *img.shape[:2])
    assert dets.masks.dtype == np.uint8
    assert set(np.unique(dets.masks)).issubset({0, 1})


@pytest.mark.skipif(not SAFE_PATH.exists(), reason="需要 weights/yolo26n-seg.safetensors")
def test_pt_engine_seg_nms_path():
    """--nms（o2m + NMS）路径同样产出掩码，且与 E2E 路径的高分检出一致"""
    import cv2

    from utils.engines import PtEngine

    img = cv2.imread(str(IMG_PATH))
    e2e = PtEngine(str(SAFE_PATH), scale="n", model="yolo26-seg", device="cpu", end2end=True).predict(img, 0.25)
    nms = PtEngine(str(SAFE_PATH), scale="n", model="yolo26-seg", device="cpu", end2end=False).predict(img, 0.25)
    assert nms.masks.shape == (len(nms.boxes), *img.shape[:2])
    assert len(nms.boxes) >= len(e2e.boxes)  # NMS 路径不做每图 300 上限的 top-k 截断
    # E2E 的高分框应能在 NMS 结果里找到同位置同类的框
    from utils.iou import box_iou

    hi = e2e.scores > 0.5
    for box, cls in zip(e2e.boxes[hi], e2e.class_ids[hi]):
        same_cls = nms.boxes[nms.class_ids == cls]
        assert len(same_cls) and box_iou(box, same_cls).max() > 0.9


@pytest.mark.skipif(not SAFE_PATH.exists(), reason="需要 weights/yolo26n-seg.safetensors")
def test_onnx_engine_seg_parity(tmp_path):
    """导出的 onnx（双输出）与 pt 引擎：框 ≈ 一致、掩码逐像素一致"""
    import cv2
    import torch

    from model.build import build_yolo26_seg
    from model.weights import load_weights
    from utils.engines import OnnxEngine, PtEngine

    model = build_yolo26_seg("n")
    load_weights(model, str(SAFE_PATH))
    onnx_path = tmp_path / "seg.onnx"
    with torch.no_grad():
        torch.onnx.export(model, torch.randn(1, 3, 640, 640), str(onnx_path), input_names=["images"],
                          output_names=["output0", "output1"], opset_version=18, dynamo=False)

    img = cv2.imread(str(IMG_PATH))
    d_onnx = OnnxEngine(str(onnx_path), scale="n", model="yolo26-seg").predict(img, 0.25)
    d_pt = PtEngine(str(SAFE_PATH), scale="n", model="yolo26-seg", device="cpu").predict(img, 0.25)
    assert d_onnx.masks.shape == d_pt.masks.shape
    assert np.abs(d_onnx.boxes - d_pt.boxes).max() < 1e-3
    assert (d_onnx.masks == d_pt.masks).all(), "onnx 与 pt 的掩码必须逐像素一致"


@pytest.mark.skipif(not (SAFE_PATH.exists() and PT_PATH.exists()), reason="需要官方 .pt 与转换后的 .safetensors")
def test_process_mask_matches_official():
    """掩码解码与官方逐像素一致：官方 process_mask(upsample) + scale_masks 对齐本实现

    这是 process_mask 唯一实现的回归闸门——阈值顺序 / 裁剪时机 / 反 letterbox 口径改动都会在此现形。
    """
    pytest.importorskip("ultralytics")
    import cv2
    import torch
    from ultralytics.utils import ops

    from model.build import build_yolo26_seg

    official_sd = torch.load(PT_PATH, map_location="cpu", weights_only=False)["model"].float().state_dict()
    model = build_yolo26_seg("n")
    model.load_state_dict(official_sd, strict=True)

    img, x, ratio, pad = _letterbox_prep()
    ori_h, ori_w = img.shape[:2]
    net_hw = (640, 640)
    with torch.no_grad():
        dets, proto = model(x)
    keep = dets[0, :, 4] > 0.25
    boxes_net, coeff = dets[0, keep][:, :4], dets[0, keep][:, 6:]

    off = ops.process_mask(proto[0], coeff, boxes_net, net_hw, upsample=True)
    off = ops.scale_masks(off[None].float(), (ori_h, ori_w),
                          ratio_pad=((ratio, ratio), (pad[1], pad[0])))[0].gt_(0.5).byte()
    ours = process_mask(proto[0].numpy(), coeff.numpy(), boxes_net.numpy(), net_hw, (ori_h, ori_w), ratio, pad)

    assert off.shape == ours.shape
    assert int((off.numpy() != ours).sum()) == 0


def test_draw_detections_overlays_masks(tmp_path):
    """可视化：带 masks 的 Detections 走掩码叠加路径（不崩、输出形状不变）"""
    from utils.engines.base import Detections
    from utils.visualize import draw_detections

    img = np.full((64, 80, 3), 128, dtype=np.uint8)
    masks = np.zeros((1, 64, 80), dtype=np.uint8)
    masks[0, 10:40, 20:60] = 1
    dets = Detections(boxes=np.array([[20, 10, 60, 40]], dtype=np.float32),
                      scores=np.array([0.9], dtype=np.float32),
                      class_ids=np.array([0]), masks=masks)
    out = draw_detections(img, dets, names={0: "x"}, save_path=str(tmp_path / "o.jpg"))
    assert out.shape == img.shape
    assert not np.array_equal(out[25, 40], img[25, 40]), "掩码区域应被着色"
