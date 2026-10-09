"""推理引擎：PtEngine / OnnxEngine / TRTEngine，统一 predict(image_bgr) -> Detections

E2E 路径：模型图内完成 top-k（输出 (300,6)），此处仅过滤置信度与还原坐标。
NMS 路径：模型输出原始 (4+nc, N)，此处做 numpy 解码 + 按类 NMS。
v3 路径：模型（V3Detect eval）输出解码 (NA, 5+nc)，此处按 obj×cls + 按类 NMS。
输入尺寸：显式参数 > 权重 metadata（safetensors header，可选）> 640（config.inference.IMGSZ）；
onnx 以图内固定 shape 为准（显式冲突报错）——以上均为可选，旧权重行为不变。
"""

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from config.inference import CONF_THRES, IMGSZ, IOU_THRES, MAX_DET
from data.preprocess import preprocess
from model.weights import apply_meta, load_meta, load_weights
from model.build import YOLO26_FAMILY, build_model
from utils.postprocess import decode_raw, non_max_suppression, scale_boxes, v3_detections

__all__ = ["Detections", "PtEngine", "OnnxEngine", "TRTEngine", "build_trt_engine",
           "resolve_device", "device_label"]

logger = logging.getLogger(__name__)


def resolve_device(device=None):
    """pt 口径的设备解析：显式给则用，否则 cuda 可用就 cuda（PtEngine 与脚本的环境行共用）"""
    return torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))


def device_label(device):
    """环境行的人类可读设备名：CUDA <型号> / CPU

    onnx 后端固定 CPUExecutionProvider（见 OnnxEngine），脚本对 onnx 传 torch.device("cpu")。
    推理脚本要在 engine 构建之前打印环境行，所以这里只吃 device 对象、不依赖 engine。
    """
    device = torch.device(device)
    return f"CUDA {torch.cuda.get_device_name(device)}" if device.type == "cuda" else "CPU"


@dataclass
class Detections:
    """检测结果（原图像素坐标）"""

    boxes: np.ndarray  # (N, 4) xyxy
    scores: np.ndarray  # (N,)
    class_ids: np.ndarray  # (N,) int


def _to_detections(out, end2end, ratio, pad, ori_shape, conf_thres, iou_thres, arch="yolo26", imgsz=None):
    """引擎输出 -> Detections（原图坐标）

    imgsz：o2m 原始输出的网格形状按它推（[(S//8)², (S//16)², (S//32)²]；640 -> 80/40/20，
    与 utils.postprocess._LEVEL_SHAPES 同口径）；None 时用模块默认（640 形状）。
    """
    h, w = ori_shape
    if arch not in YOLO26_FAMILY:
        # out: (NA, 5+nc) [x1,y1,x2,y2, obj, cls...]；最终分 = obj×cls，按类 NMS
        det = v3_detections(out, conf_thres or CONF_THRES, iou_thres, MAX_DET)
        boxes, scores, cls = det[:, :4], det[:, 4], det[:, 5].astype(np.int64)
    elif end2end:
        # out: (300, 6) [x1, y1, x2, y2, conf, cls]，图内 top-k 已完成
        mask = out[:, 4] > (conf_thres if conf_thres is not None else 0.25)
        boxes, scores, cls = out[mask][:, :4], out[mask][:, 4], out[mask][:, 5].astype(np.int64)
    else:
        # out: (4+nc, N) 原始输出（N = Σ(S//stride)²，随输入尺寸变）
        shapes = [(imgsz // s, imgsz // s) for s in (8, 16, 32)] if imgsz else None
        boxes, score_map = decode_raw(out, shapes=shapes)
        boxes, scores, cls = non_max_suppression(boxes, score_map, conf_thres or CONF_THRES, iou_thres, MAX_DET)
    boxes = scale_boxes(boxes, ratio, pad, h, w)
    return Detections(boxes, scores, cls)


class PtEngine:
    """PyTorch 权重推理

    imgsz / nc 缺省（None）时从权重 metadata 兜底——训练保存的 best/last 自带
    {nc, imgsz, anchors...}；旧权重没有 metadata 则完全按历史行为（imgsz 640 / 模型 yaml 的 nc）。
    """

    def __init__(self, weights, scale="n", end2end=True, device=None, nc=None, model="yolo26", imgsz=None):
        self.device = resolve_device(device)
        self.scale = scale
        self.arch = model
        self.end2end = end2end
        meta = load_meta(weights)  # 可选 metadata：缺失 = 空 dict
        self.imgsz = int(imgsz) if imgsz is not None else int(meta.get("imgsz") or IMGSZ)
        self.model = build_model(model, scale, imgsz=self.imgsz,
                                 nc=nc if nc is not None else meta.get("nc"))
        for key in apply_meta(self.model, meta):  # v3 anchors 随权重（yaml 为原版也能还原训练值）
            logger.info(f"weights metadata applied: {key}")
        head = self.model.model[-1]
        if hasattr(head, "end2end"):  # V3Detect 无此开关（前向即解码）
            head.end2end = end2end
        load_weights(self.model, weights, strict=True)
        self.model.to(self.device).eval()
        self._warmup()

    def _warmup(self):
        """预热：首次前向触发 CUDA kernel 编译，跑 3 次后计时才反映稳态性能（dummy 用实际输入尺寸）"""
        dummy = torch.zeros(1, 3, self.imgsz, self.imgsz, device=self.device)
        with torch.no_grad():
            for _ in range(3):
                self.model(dummy)

    @property
    def summary(self):
        """(层数, 参数量)，供输出头部摘要；层数 = 叶子模块数（官方 260 层口径）"""
        n_layers = sum(1 for m in self.model.modules() if not list(m.children()))
        return n_layers, sum(p.numel() for p in self.model.parameters())

    @property
    def summary_line(self):
        """模型行的一段：模块树口径（与训练日志同源，可与 yaml 逐层表逐项对照）"""
        n_layers, n_params = self.summary
        return f"{n_layers} layers · {n_params:,} params" if n_layers else f"{n_params:,} params"

    def predict(self, image_bgr, conf_thres=None, iou_thres=IOU_THRES):
        dets, _ = self.predict_timed(image_bgr, conf_thres, iou_thres)
        return dets

    def predict_timed(self, image_bgr, conf_thres=None, iou_thres=IOU_THRES):
        """带三段计时的预测（eval 用）：返回 (Detections, {"preprocess"/"inference"/"postprocess": ms})"""
        t0 = time.perf_counter()
        img, ratio, pad = preprocess(image_bgr, self.imgsz)
        t1 = time.perf_counter()
        with torch.no_grad():
            out = self.model(img.to(self.device))[0].cpu().numpy()
        t2 = time.perf_counter()
        dets = _to_detections(out, self.end2end, ratio, pad, image_bgr.shape[:2], conf_thres, iou_thres,
                              arch=self.arch, imgsz=self.imgsz)
        t3 = time.perf_counter()
        return dets, {"preprocess": (t1 - t0) * 1e3, "inference": (t2 - t1) * 1e3, "postprocess": (t3 - t2) * 1e3}


class OnnxEngine:
    """onnxruntime 推理（配合 scripts/export.py 导出的 onnx）

    输入尺寸以图内固定 shape 为准（`--dynamic` 只放开 batch 轴）；显式 imgsz 与图冲突时报错
    （改尺寸要重新导出），图内 shape 不可知（全动态导出）时用显式值或 640。
    """

    def __init__(self, onnx_path, scale="n", end2end=True, model="yolo26", imgsz=None):
        self.scale = scale
        self.arch = model
        self.end2end = end2end
        self.onnx_path = onnx_path
        self.sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name
        shape = self.sess.get_inputs()[0].shape  # (1, 3, H, W)
        graph_imgsz = shape[2] if len(shape) == 4 and isinstance(shape[2], int) else None
        if imgsz is not None and graph_imgsz is not None and int(imgsz) != graph_imgsz:
            raise ValueError(f"onnx graph is fixed at {graph_imgsz}px — --imgsz {imgsz} conflicts "
                             f"(re-export with --imgsz {imgsz} to use that size)")
        self.imgsz = graph_imgsz or (int(imgsz) if imgsz is not None else IMGSZ)
        self._warmup()

    def _warmup(self):
        """预热：跑 3 次后计时才反映稳态性能（dummy 用实际输入尺寸）"""
        dummy = np.zeros((1, 3, self.imgsz, self.imgsz), dtype=np.float32)
        for _ in range(3):
            self.sess.run(None, {self.input_name: dummy})

    @property
    def summary(self):
        """(层数, 参数量)：onnx 无层数概念，只统计 initializer 参数量"""
        import onnx

        graph = onnx.load(self.onnx_path, load_external_data=False).graph
        return None, sum(int(np.prod(t.dims)) for t in graph.initializer)

    @property
    def summary_line(self):
        """模型行的一段：部署口径

        ONNX 图没有模块层级——导出时 Conv+BN 已折叠、SiLU 保持 Sigmoid+Mul，`initializer`
        求和得到的"参数量"与 pt 侧 `parameters()` 不是同一个量（本机 yolo26n: 2,408,932 vs
        2,572,280，差的就是被折叠的 BN），故此处只报文件大小与 I/O 形状；要看逐层结构用 netron。
        """
        mib = Path(self.onnx_path).stat().st_size / 2**20
        in_shape = tuple(self.sess.get_inputs()[0].shape)
        out_shape = tuple(self.sess.get_outputs()[0].shape)
        return f"ONNX {mib:.1f} MiB · in {in_shape} · out {out_shape}"

    def predict(self, image_bgr, conf_thres=None, iou_thres=IOU_THRES):
        dets, _ = self.predict_timed(image_bgr, conf_thres, iou_thres)
        return dets

    def predict_timed(self, image_bgr, conf_thres=None, iou_thres=IOU_THRES):
        """带三段计时的预测（eval 用）：返回 (Detections, {"preprocess"/"inference"/"postprocess": ms})"""
        t0 = time.perf_counter()
        img, ratio, pad = preprocess(image_bgr, self.imgsz)
        t1 = time.perf_counter()
        out = self.sess.run(None, {self.input_name: img.numpy()})[0][0]
        t2 = time.perf_counter()
        dets = _to_detections(out, self.end2end, ratio, pad, image_bgr.shape[:2], conf_thres, iou_thres,
                              arch=self.arch, imgsz=self.imgsz)
        t3 = time.perf_counter()
        return dets, {"preprocess": (t1 - t0) * 1e3, "inference": (t2 - t1) * 1e3, "postprocess": (t3 - t2) * 1e3}


def _trt_torch_dtype(trt, engine, name):
    """TRT tensor dtype -> torch dtype（经 numpy 中转）"""
    return torch.from_numpy(np.empty(1, dtype=trt.nptype(engine.get_tensor_dtype(name)))).dtype


def build_trt_engine(onnx_path, engine_path, fp16=False, workspace_gb=1):
    """onnx -> TensorRT 序列化 engine（scripts/export.py --trt 的底层实现）

    engine 与构建机的 GPU 型号 / TRT 版本绑定（TRT 固有约束，不跨机移植）——换环境需重新构建。
    tensorrt 是可选依赖：懒加载，未安装时报 ImportError（调用方转成用户可读的提示）。
    """
    try:
        import tensorrt as trt
    except ImportError as e:
        raise ImportError(
            "TensorRT requires the optional dependency 'tensorrt' "
            "(pip install tensorrt-cu13, see https://docs.nvidia.com/deeplearning/tensorrt/)") from e
    trt_logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(trt_logger)
    try:  # TRT 10 及更早需要显式 EXPLICIT_BATCH 标志；TRT 11 移除该标志（已是唯一模式）
        flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    except AttributeError:
        flags = 0
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, trt_logger)
    if not parser.parse_from_file(str(onnx_path)):
        errors = "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"TensorRT failed to parse {onnx_path}: {errors}")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)
    if fp16:
        fp16_flag = getattr(trt.BuilderFlag, "FP16", None)
        if fp16_flag is None:  # TRT 11+ 移除 FP16 标志（精度由 strongly-typed 网络控制）
            raise RuntimeError("--fp16 is not supported by this TensorRT version (>= 11 removed the FP16 "
                               "builder flag; precision is controlled by strongly-typed networks) — "
                               "rebuild without --fp16")
        config.set_flag(fp16_flag)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(f"TensorRT engine build failed for {onnx_path}")
    Path(engine_path).parent.mkdir(parents=True, exist_ok=True)
    Path(engine_path).write_bytes(bytes(serialized))


class TRTEngine:
    """TensorRT 推理（配合 scripts/export.py --trt 构建的 .engine）

    engine 与构建它的 GPU/TRT 版本绑定——换环境需重新构建（TRT 固有约束）。
    输入尺寸以 engine 内固定 shape 为准（显式 imgsz 冲突报错）；仅支持静态 shape 的 engine。
    tensorrt 是可选依赖：懒加载，未安装时报 ImportError。
    """

    def __init__(self, engine_path, scale="n", end2end=True, model="yolo26", imgsz=None):
        try:
            import tensorrt as trt
        except ImportError as e:
            raise ImportError(
                "TensorRT engine inference requires the optional dependency 'tensorrt' "
                "(pip install tensorrt-cu13, see https://docs.nvidia.com/deeplearning/tensorrt/)") from e
        if not torch.cuda.is_available():
            raise RuntimeError("TensorRT engine requires a CUDA device")
        self._trt = trt
        self.scale = scale
        self.arch = model
        self.end2end = end2end
        self.engine_path = engine_path
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        engine = runtime.deserialize_cuda_engine(Path(engine_path).read_bytes())
        if engine is None:
            raise ValueError(f"failed to deserialize TensorRT engine: {engine_path}")
        self.engine = engine
        self.ctx = engine.create_execution_context()
        names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
        ins = [n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        outs = [n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
        if len(ins) != 1 or len(outs) != 1:
            raise ValueError(f"expected 1 input / 1 output in engine, got {ins} / {outs}")
        self.input_name, self.output_name = ins[0], outs[0]
        self.in_shape = tuple(engine.get_tensor_shape(self.input_name))
        self.out_shape = tuple(engine.get_tensor_shape(self.output_name))
        if any(d < 0 for d in self.in_shape + self.out_shape):
            raise ValueError(f"dynamic shapes are not supported (in {self.in_shape}, out {self.out_shape}); "
                             f"build the engine from a fixed-shape onnx (no --dynamic)")
        graph_imgsz = self.in_shape[2] if len(self.in_shape) == 4 else None
        if imgsz is not None and graph_imgsz is not None and int(imgsz) != graph_imgsz:
            raise ValueError(f"TRT engine is fixed at {graph_imgsz}px — --imgsz {imgsz} conflicts "
                             f"(rebuild the engine for that size)")
        self.imgsz = graph_imgsz or (int(imgsz) if imgsz is not None else IMGSZ)
        self._in = torch.empty(self.in_shape, dtype=_trt_torch_dtype(trt, engine, self.input_name), device="cuda")
        self._out = torch.empty(self.out_shape, dtype=_trt_torch_dtype(trt, engine, self.output_name), device="cuda")
        self._stream = torch.cuda.Stream()  # 专用流：TRT 建议非默认流（默认流会在 enqueue 内部加同步）
        self.ctx.set_tensor_address(self.input_name, self._in.data_ptr())
        self.ctx.set_tensor_address(self.output_name, self._out.data_ptr())
        self._warmup()

    def _run(self, img_np):
        """执行一次：H2D 拷贝 -> execute_async_v3（专用流）-> 流同步后读回（阻塞，与 ORT .run 同语义）"""
        with torch.cuda.stream(self._stream):
            self._in.copy_(torch.from_numpy(img_np))
            self.ctx.execute_async_v3(self._stream.cuda_stream)
        self._stream.synchronize()
        return self._out.cpu().numpy()

    def _warmup(self):
        """预热：跑 3 次后计时才反映稳态性能（dummy 用实际输入尺寸）"""
        dummy = np.zeros(self.in_shape, dtype=np.float32)
        for _ in range(3):
            self._run(dummy)

    @property
    def summary_line(self):
        """模型行的一段：部署口径（engine 文件大小 + I/O 形状；逐层结构见构建它的 onnx）"""
        mib = Path(self.engine_path).stat().st_size / 2**20
        return f"TensorRT {mib:.1f} MiB · in {self.in_shape} · out {self.out_shape}"

    def predict(self, image_bgr, conf_thres=None, iou_thres=IOU_THRES):
        dets, _ = self.predict_timed(image_bgr, conf_thres, iou_thres)
        return dets

    def predict_timed(self, image_bgr, conf_thres=None, iou_thres=IOU_THRES):
        """带三段计时的预测（eval 用）：返回 (Detections, {"preprocess"/"inference"/"postprocess": ms})"""
        t0 = time.perf_counter()
        img, ratio, pad = preprocess(image_bgr, self.imgsz)
        t1 = time.perf_counter()
        out = self._run(img.numpy())[0]
        t2 = time.perf_counter()
        dets = _to_detections(out, self.end2end, ratio, pad, image_bgr.shape[:2], conf_thres, iou_thres,
                              arch=self.arch, imgsz=self.imgsz)
        t3 = time.perf_counter()
        return dets, {"preprocess": (t1 - t0) * 1e3, "inference": (t2 - t1) * 1e3, "postprocess": (t3 - t2) * 1e3}
