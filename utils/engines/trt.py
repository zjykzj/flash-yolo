"""TensorRT 后端：TRTEngine 运行时 + build_trt_engine 构建器（onnx -> .engine）

engine 与构建机的 GPU 型号 / TRT 版本绑定（TRT 固有约束，不跨机移植）——换环境需重新构建。
tensorrt 是可选依赖：懒加载，未安装时报 ImportError。
"""

from pathlib import Path

import numpy as np
import torch

from config.inference import IMGSZ
from utils.engines.base import BaseEngine

__all__ = ["TRTEngine", "build_trt_engine"]


def _trt_torch_dtype(trt, engine, name):
    """TRT tensor dtype -> torch dtype（经 numpy 中转）"""
    return torch.from_numpy(np.empty(1, dtype=trt.nptype(engine.get_tensor_dtype(name)))).dtype


def build_trt_engine(onnx_path, engine_path, fp16=False, workspace_gb=1):
    """onnx -> TensorRT 序列化 engine（scripts/export.py --trt 的底层实现）

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


class TRTEngine(BaseEngine):
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

    def _forward(self, img):
        """单图前向：H2D 拷贝 -> execute_async_v3（专用流）-> 流同步后读回（阻塞，与 ORT .run 同语义）"""
        with torch.cuda.stream(self._stream):
            self._in.copy_(img)
            self.ctx.execute_async_v3(self._stream.cuda_stream)
        self._stream.synchronize()
        return self._out[0].cpu().numpy()  # 剥批：引擎输出 (1, ...)，契约与 PtEngine / OnnxEngine 一致

    def _warmup(self):
        """预热：跑 3 次后计时才反映稳态性能（dummy 用实际输入尺寸）"""
        dummy = torch.zeros(self.in_shape, dtype=torch.float32)
        for _ in range(3):
            self._forward(dummy)

    @property
    def summary_line(self):
        """模型行的一段：部署口径（engine 文件大小 + I/O 形状；逐层结构见构建它的 onnx）"""
        mib = Path(self.engine_path).stat().st_size / 2**20
        return f"TensorRT {mib:.1f} MiB · in {self.in_shape} · out {self.out_shape}"
