"""pt (safetensors) -> onnx 导出

yolo26：默认导出 E2E 融合图，单输出 output0 (B, 300, 6)，解码与两阶段 top-k 全部在图内；
--raw 导出原始头输出 (B, 4+nc, 8400)，配合 NMS 路径在外部解码（utils/postprocess.py）。
yolov3-tiny：单输出解码 (B, NA, 5+nc)——obj×cls 与按类 NMS 在外部（utils/postprocess.v3_detections），
--raw 不适用。
默认 batch 固定为 1（边缘工具链偏好固定 shape），--dynamic 打开动态 batch。

用法:
    python scripts/export.py --weights weights/yolo26n.safetensors --out weights/yolo26n.onnx
    python scripts/export.py --weights weights/yolo26n.safetensors --out weights/yolo26n_raw.onnx --raw
    python scripts/export.py --weights weights/yolov3-tiny.safetensors --model yolov3-tiny
    python scripts/export.py --weights weights/yolo26n.safetensors --trt [--fp16]   # 顺带构建 TensorRT engine

--trt：onnx 导出后继续构建序列化 engine（与构建机 GPU/TRT 版本绑定、不跨机移植；换机需重构建）。
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # 仓库根目录入 sys.path

import torch

from model.weights import load_weights, resolve_arch_scale, resolve_imgsz
from model.build import ARCHS, YOLO26_FAMILY, build_model
from utils.logger import get_logger, log_params, setup_logging

setup_logging()
logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="safetensors -> onnx")
    parser.add_argument("--weights", required=True, help=".safetensors path")
    parser.add_argument("--out", default=None, help="output .onnx path (default runs/export/<weights>.onnx)")
    parser.add_argument("--raw", action="store_true", help="export raw head output (yolo26 only)")
    parser.add_argument("--dynamic", action="store_true", help="dynamic batch axis (default: fixed batch=1)")
    parser.add_argument("--model", default=None, choices=sorted(ARCHS),
                        help="architecture (default: inferred from the weights filename)")
    parser.add_argument("--scale", default="n", help="model scale (yolo26 only: n/s/m/l/x)")
    parser.add_argument("--nc", type=int, default=None,
                        help="class count for models trained on a non-COCO dataset "
                             "(default: the model yaml's nc)")
    parser.add_argument("--imgsz", type=int, default=None,
                        help="export input size (default: weights metadata, else 640)")
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--trt", action="store_true",
                        help="also build a TensorRT engine (.engine) from the exported onnx (needs tensorrt + CUDA)")
    parser.add_argument("--fp16", action="store_true", help="build the TRT engine with fp16 precision (requires --trt)")
    args = parser.parse_args()

    arch, scale = resolve_arch_scale(args.weights, args.model, args.scale)
    if arch is None:
        parser.error(f"cannot infer the model from '{args.weights}' — pass --model and/or --scale")
    if arch == "yolo26" and scale is None:
        parser.error(f"cannot infer the model scale from '{args.weights}' — pass --scale n/s/m/l/x")
    if arch not in YOLO26_FAMILY and args.raw:
        parser.error(f"--raw is only supported for yolo26 (got model={arch!r})")
    if args.fp16 and not args.trt:
        parser.error("--fp16 requires --trt")
    if args.trt and args.dynamic:
        parser.error("--trt requires fixed shapes (drop --dynamic)")
    imgsz = resolve_imgsz(args.weights, args.imgsz)  # CLI > 权重 metadata > 640

    out_path = args.out or str(ROOT / "runs" / "export" / (Path(args.weights).stem + ".onnx"))
    log_params(logger, __file__, weights=args.weights, out=out_path, model=arch, imgsz=imgsz,
               nc=args.nc if args.nc is not None else 80, opset=args.opset,
               dynamic=args.dynamic, raw=args.raw, trt=args.trt, fp16=args.fp16)

    model = build_model(arch, scale, nc=args.nc)
    load_weights(model, args.weights, strict=True)
    head = model.model[-1]
    if hasattr(head, "end2end"):  # V3Detect 无此开关（前向即解码）
        head.end2end = not args.raw
    model.eval()

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    dummy = torch.randn(1, 3, imgsz, imgsz)
    # 默认固定 batch=1（边缘工具链偏好固定 shape）；--dynamic 打开动态 batch
    dynamic_axes = {"images": {0: "batch"}, "output0": {0: "batch"}} if args.dynamic else None
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy,
            out_path,
            input_names=["images"],
            output_names=["output0"],
            dynamic_axes=dynamic_axes,
            opset_version=args.opset,
            dynamo=False,  # TorchScript 导出器：权重内嵌单文件；dynamo 默认把权重拆到 .onnx.data
        )
    mode = ("raw NMS path" if args.raw else "E2E fused") if arch in YOLO26_FAMILY else "decoded (v3)"
    logger.info(f"exported ({mode}) -> {out_path}")

    if args.trt:
        from utils.engine import build_trt_engine  # 懒加载：非 trt 路径不引入 engine 依赖

        engine_path = Path(out_path).with_suffix(".engine")
        try:
            build_trt_engine(out_path, engine_path, fp16=args.fp16)
        except (ImportError, RuntimeError) as e:
            parser.error(str(e))
        mib = Path(engine_path).stat().st_size / 2**20
        logger.info(f"exported TensorRT engine ({'fp16' if args.fp16 else 'fp32'}) -> {engine_path} "
                    f"({mib:.1f} MiB; bound to this GPU + TRT version)")


if __name__ == "__main__":
    main()
