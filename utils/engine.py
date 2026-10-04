"""推理引擎：PtEngine / OnnxEngine，统一 predict(image_bgr) -> Detections

E2E 路径：模型图内完成 top-k（输出 (300,6)），此处仅过滤置信度与还原坐标。
NMS 路径：模型输出原始 (4+nc, 8400)，此处做 numpy 解码 + 按类 NMS。
"""

import time
from dataclasses import dataclass

import numpy as np
import onnxruntime as ort
import torch

from config.defaults import CONF_THRES, IOU_THRES, MAX_DET
from data.preprocess import preprocess
from model.weights import load_weights
from model.yolo26 import build_yolo26
from utils.postprocess import decode_raw, non_max_suppression, scale_boxes

__all__ = ["Detections", "PtEngine", "OnnxEngine"]


@dataclass
class Detections:
    """检测结果（原图像素坐标）"""

    boxes: np.ndarray  # (N, 4) xyxy
    scores: np.ndarray  # (N,)
    class_ids: np.ndarray  # (N,) int


def _to_detections(out, end2end, ratio, pad, ori_shape, conf_thres, iou_thres):
    """引擎输出 -> Detections（原图坐标）"""
    h, w = ori_shape
    if end2end:
        # out: (300, 6) [x1, y1, x2, y2, conf, cls]，图内 top-k 已完成
        mask = out[:, 4] > (conf_thres if conf_thres is not None else 0.25)
        boxes, scores, cls = out[mask][:, :4], out[mask][:, 4], out[mask][:, 5].astype(np.int64)
    else:
        # out: (4+nc, 8400) 原始输出
        boxes, score_map = decode_raw(out)
        boxes, scores, cls = non_max_suppression(boxes, score_map, conf_thres or CONF_THRES, iou_thres, MAX_DET)
    boxes = scale_boxes(boxes, ratio, pad, h, w)
    return Detections(boxes, scores, cls)


class PtEngine:
    """PyTorch 权重推理"""

    def __init__(self, weights, scale="n", end2end=True, device=None):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)
        self.scale = scale
        self.end2end = end2end
        self.model = build_yolo26(scale)
        self.model.model[-1].end2end = end2end
        load_weights(self.model, weights, strict=True)
        self.model.to(self.device).eval()
        self._warmup()

    def _warmup(self):
        """预热：首次前向触发 CUDA kernel 编译，跑 3 次后计时才反映稳态性能"""
        dummy = torch.zeros(1, 3, 640, 640, device=self.device)
        with torch.no_grad():
            for _ in range(3):
                self.model(dummy)

    @property
    def summary(self):
        """(层数, 参数量)，供输出头部摘要；层数 = 叶子模块数（官方 260 层口径）"""
        n_layers = sum(1 for m in self.model.modules() if not list(m.children()))
        return n_layers, sum(p.numel() for p in self.model.parameters())

    def predict(self, image_bgr, conf_thres=None, iou_thres=IOU_THRES):
        dets, _ = self.predict_timed(image_bgr, conf_thres, iou_thres)
        return dets

    def predict_timed(self, image_bgr, conf_thres=None, iou_thres=IOU_THRES):
        """带三段计时的预测（eval 用）：返回 (Detections, {"preprocess"/"inference"/"postprocess": ms})"""
        t0 = time.perf_counter()
        img, ratio, pad = preprocess(image_bgr)
        t1 = time.perf_counter()
        with torch.no_grad():
            out = self.model(img.to(self.device))[0].cpu().numpy()
        t2 = time.perf_counter()
        dets = _to_detections(out, self.end2end, ratio, pad, image_bgr.shape[:2], conf_thres, iou_thres)
        t3 = time.perf_counter()
        return dets, {"preprocess": (t1 - t0) * 1e3, "inference": (t2 - t1) * 1e3, "postprocess": (t3 - t2) * 1e3}


class OnnxEngine:
    """onnxruntime 推理（配合 scripts/export.py 导出的 onnx）"""

    def __init__(self, onnx_path, scale="n", end2end=True):
        self.scale = scale
        self.end2end = end2end
        self.onnx_path = onnx_path
        self.sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name
        self._warmup()

    def _warmup(self):
        """预热：跑 3 次后计时才反映稳态性能"""
        dummy = np.zeros((1, 3, 640, 640), dtype=np.float32)
        for _ in range(3):
            self.sess.run(None, {self.input_name: dummy})

    @property
    def summary(self):
        """(层数, 参数量)：onnx 无层数概念，只统计 initializer 参数量"""
        import onnx

        graph = onnx.load(self.onnx_path, load_external_data=False).graph
        return None, sum(int(np.prod(t.dims)) for t in graph.initializer)

    def predict(self, image_bgr, conf_thres=None, iou_thres=IOU_THRES):
        dets, _ = self.predict_timed(image_bgr, conf_thres, iou_thres)
        return dets

    def predict_timed(self, image_bgr, conf_thres=None, iou_thres=IOU_THRES):
        """带三段计时的预测（eval 用）：返回 (Detections, {"preprocess"/"inference"/"postprocess": ms})"""
        t0 = time.perf_counter()
        img, ratio, pad = preprocess(image_bgr)
        t1 = time.perf_counter()
        out = self.sess.run(None, {self.input_name: img.numpy()})[0][0]
        t2 = time.perf_counter()
        dets = _to_detections(out, self.end2end, ratio, pad, image_bgr.shape[:2], conf_thres, iou_thres)
        t3 = time.perf_counter()
        return dets, {"preprocess": (t1 - t0) * 1e3, "inference": (t2 - t1) * 1e3, "postprocess": (t3 - t2) * 1e3}
