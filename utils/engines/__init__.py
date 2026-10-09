"""推理引擎包：Pt / Onnx(CPU) / TRT(GPU) 三后端，统一 predict(image_bgr) -> Detections

拆包成例照 model/basic/（按类型分文件 + 本文件统一出口）；对外名字与旧的 utils.engine 模块一致：

    from utils.engines import PtEngine, OnnxEngine, TRTEngine, build_trt_engine, ...
"""

from utils.engines.base import BaseEngine, Detections, device_label, resolve_device
from utils.engines.onnx import OnnxEngine
from utils.engines.pt import PtEngine
from utils.engines.trt import TRTEngine, build_trt_engine

__all__ = ["BaseEngine", "Detections", "PtEngine", "OnnxEngine", "TRTEngine", "build_trt_engine",
           "resolve_device", "device_label"]
