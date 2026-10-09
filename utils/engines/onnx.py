"""ONNX Runtime 后端：OnnxEngine（固定 CPUExecutionProvider；配合 scripts/export.py 的 onnx）"""

from pathlib import Path

import numpy as np
import onnxruntime as ort

from config.inference import IMGSZ
from utils.engines.base import BaseEngine

__all__ = ["OnnxEngine"]


class OnnxEngine(BaseEngine):
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

    def _forward(self, img):
        return self.sess.run(None, {self.input_name: img.numpy()})[0][0]

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
