"""pt (safetensors) -> onnx 导出

默认导出 E2E 融合图：单输出 output0 (B, 300, 6)，解码与两阶段 top-k 全部在图内；
--raw 导出原始头输出 (B, 4+nc, 8400)，配合 NMS 路径在外部解码（utils/postprocess.py）。
默认 batch 固定为 1（边缘工具链偏好固定 shape），--dynamic 打开动态 batch。

用法:
    python scripts/export.py --weights weights/yolo26n.safetensors --out weights/yolo26n.onnx
    python scripts/export.py --weights weights/yolo26n.safetensors --out weights/yolo26n_raw.onnx --raw
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # 仓库根目录入 sys.path

import torch

from model.weights import load_weights
from model.yolo26 import build_yolo26
from utils.logger import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="safetensors -> onnx")
    parser.add_argument("--weights", required=True, help=".safetensors path")
    parser.add_argument("--out", default=None, help="output .onnx path (default runs/export/<weights>.onnx)")
    parser.add_argument("--raw", action="store_true", help="export raw head output (for NMS path)")
    parser.add_argument("--dynamic", action="store_true", help="dynamic batch axis (default: fixed batch=1)")
    parser.add_argument("--scale", default="n", help="model scale (n/s/m/l/x)")
    parser.add_argument("--nc", type=int, default=None,
                        help="class count for models trained on a non-COCO dataset "
                             "(default: the model yaml's nc)")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--opset", type=int, default=18)
    args = parser.parse_args()

    model = build_yolo26(args.scale, nc=args.nc)
    load_weights(model, args.weights, strict=True)
    model.model[-1].end2end = not args.raw
    model.eval()

    out_path = args.out or str(ROOT / "runs" / "export" / (Path(args.weights).stem + ".onnx"))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    dummy = torch.randn(1, 3, args.imgsz, args.imgsz)
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
    logger.info(f"exported ({'raw NMS path' if args.raw else 'E2E fused'}) -> {out_path}")


if __name__ == "__main__":
    main()
