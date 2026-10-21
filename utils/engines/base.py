"""引擎公共件：Detections 契约 / 解码路由 / 设备工具 / BaseEngine 骨架

E2E 路径：模型图内完成 top-k（输出 (300,6)），此处仅过滤置信度与还原坐标。
NMS 路径：模型输出原始 (4+nc, N)，此处做 numpy 解码 + 按类 NMS（utils/postprocess.py）。
v3 路径：模型（V3Detect eval）输出解码 (NA, 5+nc)，此处按 obj×cls + 按类 NMS。

BaseEngine 管 predict / predict_timed 的共同骨架（preprocess -> 子类 _forward -> 解码 -> 三段计时）。
子类契约：
    __init__：构建后端，并设置 self.imgsz / self.arch / self.end2end
    _forward(img)：单图前向，img = preprocess 输出（torch CPU 张量），返回已剥批的原始输出（ndarray）
"""

import time
from dataclasses import dataclass

import numpy as np
import torch

from config.inference import CONF_THRES, IMGSZ, IOU_THRES, MAX_DET
from data.preprocess import preprocess
from model.build import YOLO26_FAMILY
from utils.postprocess import decode_raw, non_max_suppression, process_mask, scale_boxes, v3_detections

__all__ = ["Detections", "BaseEngine", "resolve_device", "device_label"]


def resolve_device(device=None):
    """pt 口径的设备解析：显式给则用，否则 cuda 可用就 cuda（PtEngine 与脚本的环境行共用）"""
    return torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))


def device_label(device):
    """环境行的人类可读设备名：CUDA <型号> / CPU

    onnx 后端固定 CPUExecutionProvider（OnnxEngine）、trt 后端固定 CUDA（TRTEngine），脚本对 onnx
    传 torch.device("cpu")、对 trt 传 "cuda"。推理脚本要在 engine 构建之前打印环境行，所以这里
    只吃 device 对象、不依赖 engine。
    """
    device = torch.device(device)
    return f"CUDA {torch.cuda.get_device_name(device)}" if device.type == "cuda" else "CPU"


@dataclass
class Detections:
    """检测结果（原图像素坐标）

    masks 仅分割任务填写：(N, H, W) uint8 二值掩码 @原图分辨率（可视化直用；
    评估侧逐图编码 RLE 后即弃，不跨图累积）。
    """

    boxes: np.ndarray  # (N, 4) xyxy
    scores: np.ndarray  # (N,)
    class_ids: np.ndarray  # (N,) int
    masks: np.ndarray | None = None  # (N, H, W) uint8（分割）


def _to_detections(out, end2end, ratio, pad, ori_shape, conf_thres, iou_thres, arch="yolo26", imgsz=None,
                   task="detect", mask_device=None):
    """引擎输出 -> Detections（原图坐标）

    imgsz：o2m 原始输出的网格形状按它推（[(S//8)², (S//16)², (S//32)²]；640 -> 80/40/20，
    与 utils.postprocess._LEVEL_SHAPES 同口径）；None 时用模块默认（640 形状）。

    task="segment"：out = (检测输出, 原型)，检测部分的解码口径与 detect 完全一致，掩码由
    系数 × 原型解码到原图分辨率（复用同一次框选择——E2E 的系数在图内已随 top-k 取回，
    NMS 路径经 extra 透传）。
    """
    h, w = ori_shape
    proto = None
    if task == "segment":
        out, proto = out
    coeff = None
    if arch not in YOLO26_FAMILY:
        # out: (NA, 5+nc) [x1,y1,x2,y2, obj, cls...]；最终分 = obj×cls，按类 NMS
        det = v3_detections(out, conf_thres or CONF_THRES, iou_thres, MAX_DET)
        boxes, scores, cls = det[:, :4], det[:, 4], det[:, 5].astype(np.int64)
    elif end2end:
        # out: (300, 6 [+ nm]) [x1, y1, x2, y2, conf, cls, 系数...]，图内 top-k 已完成
        mask = out[:, 4] > (conf_thres if conf_thres is not None else 0.25)
        boxes, scores, cls = out[mask][:, :4], out[mask][:, 4], out[mask][:, 5].astype(np.int64)
        if task == "segment":
            coeff = out[mask][:, 6:]
    else:
        # out: (4+nc [+ nm], N) 原始输出（N = Σ(S//stride)²，随输入尺寸变）
        shapes = [(imgsz // s, imgsz // s) for s in (8, 16, 32)] if imgsz else None
        nm = proto.shape[0] if task == "segment" else 0
        nc = out.shape[0] - 4 - nm
        boxes, score_map = decode_raw(out[: 4 + nc], shapes=shapes)
        if task == "segment":
            boxes, scores, cls, coeff = non_max_suppression(
                boxes, score_map, conf_thres or CONF_THRES, iou_thres, MAX_DET, extra=out[4 + nc:].T)
        else:
            boxes, scores, cls = non_max_suppression(boxes, score_map, conf_thres or CONF_THRES, iou_thres, MAX_DET)
    masks = None
    if task == "segment":
        masks = process_mask(proto, coeff, boxes, (imgsz or IMGSZ, imgsz or IMGSZ), (h, w), ratio, pad,
                             device=mask_device)
    boxes = scale_boxes(boxes, ratio, pad, h, w)
    return Detections(boxes, scores, cls, masks)


class BaseEngine:
    """引擎共同骨架：predict / predict_timed（三段计时：preprocess / inference / postprocess）

    输入尺寸语义（各后端的求值顺序见子类 docstring）：显式参数 > 权重 metadata / 图内 shape > 640。
    """

    imgsz = IMGSZ  # 子类 __init__ 覆盖
    arch = "yolo26"
    task = "detect"  # 子类 __init__ 按 arch 覆盖（model.build.arch_task）
    end2end = True

    def _forward(self, img):
        """单图前向（子类实现）：img = preprocess 输出（torch CPU 张量）-> 已剥批的原始输出 ndarray"""
        raise NotImplementedError

    def predict(self, image_bgr, conf_thres=None, iou_thres=IOU_THRES):
        dets, _ = self.predict_timed(image_bgr, conf_thres, iou_thres)
        return dets

    def predict_timed(self, image_bgr, conf_thres=None, iou_thres=IOU_THRES):
        """带三段计时的预测（eval 用）：返回 (Detections, {"preprocess"/"inference"/"postprocess": ms})"""
        t0 = time.perf_counter()
        img, ratio, pad = preprocess(image_bgr, self.imgsz)
        t1 = time.perf_counter()
        out = self._forward(img)
        t2 = time.perf_counter()
        dets = _to_detections(out, self.end2end, ratio, pad, image_bgr.shape[:2], conf_thres, iou_thres,
                              arch=self.arch, imgsz=self.imgsz, task=self.task,
                              mask_device=getattr(self, "mask_device", None))
        t3 = time.perf_counter()
        return dets, {"preprocess": (t1 - t0) * 1e3, "inference": (t2 - t1) * 1e3, "postprocess": (t3 - t2) * 1e3}
