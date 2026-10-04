"""官方 .pt -> 纯 state_dict（safetensors）

官方 checkpoint 的 ckpt["ema"] 是 pickle 的 ModelEMA 模块对象（fp16），
反序列化需要安装 ultralytics（仅 dev-time，见 requirements-dev.txt）。
转换产物为纯权重字典，运行时（engine/tests）零 ultralytics 依赖。

用法:
    python scripts/convert_weights.py --src yolo26n.pt --dst weights/yolo26n.safetensors
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 仓库根目录入 sys.path

import torch
from safetensors.torch import save_file

from utils.logger import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="official .pt -> pure safetensors weights")
    parser.add_argument("--src", required=True, help="official .pt path")
    parser.add_argument("--dst", required=True, help="output .safetensors path")
    args = parser.parse_args()

    ckpt = torch.load(args.src, map_location="cpu", weights_only=False)
    ema = ckpt.get("ema")
    if ema is not None:
        model = ema.ema if hasattr(ema, "ema") else ema  # 旧格式：ModelEMA.ema 为模型本体
    else:
        model = ckpt["model"]  # 当前格式：ckpt["model"] 直接是完整模型对象
    sd = model.float().state_dict()
    save_file(sd, args.dst)

    keys = list(sd.keys())
    n_params = sum(v.numel() for v in sd.values())
    logger.info(f"saved {len(keys)} tensors, {n_params:,} params -> {args.dst}")
    logger.info(f"key first : {keys[0]}")
    logger.info(f"key middle: {keys[len(keys) // 2]}")
    logger.info(f"key last  : {keys[-1]}")


if __name__ == "__main__":
    main()
