"""官方权重 -> 纯 state_dict（safetensors），按扩展名分派

- ultralytics .pt（yolo26 系）：ckpt["ema"]/ckpt["model"] 解包（反序列化需安装 ultralytics，
  仅 dev-time，见 requirements-dev.txt）。
- darknet .weights（yolov3-tiny）：定长二进制逐块解析。张量序 = darknet cfg 层序（P5 头卷积
  位于 P4 支路卷积之前，即 cfg 里第一个 [yolo] 的位置）；块内顺序 [bias, scale, mean, var,
  conv 权重]（无 BN 的输出层为 [bias, 权重]）；conv 布局 (out, in, kh, kw) 与 torch 一致，
  无需转置；float32 小端。头部为 4 或 5 个 int32（版本 + seen），按文件长度自动判别，
  要求文件恰好用尽（多/少字节都报错）。

转换产物为纯权重字典，运行时（engine/tests）零 ultralytics/darknet 依赖。

用法:
    python scripts/convert_weights.py --src yolo26n.pt --dst weights/yolo26n.safetensors
    python scripts/convert_weights.py --src yolov3-tiny.weights --dst weights/yolov3-tiny.safetensors
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 仓库根目录入 sys.path

import numpy as np
import torch
from safetensors.torch import save_file

from model.build import build_model
from utils.logger import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)


def _convert_pt(src, dst):
    """ultralytics .pt -> safetensors（ema/model 解包，全键原样落盘）"""
    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    ema = ckpt.get("ema")
    if ema is not None:
        model = ema.ema if hasattr(ema, "ema") else ema  # 旧格式：ModelEMA.ema 为模型本体
    else:
        model = ckpt["model"]  # 当前格式：ckpt["model"] 直接是完整模型对象
    sd = model.float().state_dict()
    save_file(sd, dst)
    return sd


def _darknet_specs(model):
    """yolov3-tiny darknet .weights 的张量消费序（= darknet cfg 层序）

    每项 (module, kind)："bn" = 文件块 [bias(β), scale(γ), mean, var, conv 权重]；
    "plain" = [conv bias, conv 权重]（输出层无 BN）。P5 头（512→255）在文件中位于
    conv128 / conv256（P4 支路）之前——正是 cfg 第二个 [yolo] 的位置。
    层索引与 config/models/yolov3-tiny.yaml 一一对应（改 yaml 需同步此表）。
    """
    layers = list(model.model)
    head = layers[-1]
    return [
        (layers[0], "bn"), (layers[2], "bn"), (layers[4], "bn"), (layers[6], "bn"),
        (layers[8], "bn"), (layers[10], "bn"),
        (layers[13], "bn"), (layers[14], "bn"), (layers[15], "bn"),
        (head.m[1], "plain"),
        (layers[16], "bn"), (layers[19], "bn"),
        (head.m[0], "plain"),
    ]


def _convert_darknet(src, dst, nc=80):
    """darknet .weights -> safetensors（yolov3-tiny）

    键集/张量形状以空模型的 state_dict 为单一来源（num_batches_tracked 等非文件张量取零
    初值——推理不受影响，微调时该计数在固定 momentum 下不参与计算）。
    """
    model = build_model("yolov3-tiny", nc=nc)
    sd = dict(model.state_dict())
    names = {id(m): n for n, m in model.named_modules()}

    specs = _darknet_specs(model)
    n_floats, plan = 0, []  # (prefix, kind, conv 形状, 输出通道)
    for module, kind in specs:
        conv = module.conv if hasattr(module, "conv") else module
        plan.append((names[id(module)], kind, tuple(conv.weight.shape), conv.out_channels))
        n_floats += (4 * conv.out_channels if kind == "bn" else conv.out_channels) + int(np.prod(conv.weight.shape))

    data = Path(src).read_bytes()
    header = len(data) - 4 * n_floats
    if header not in (16, 20):
        raise ValueError(f"file size {len(data)} does not match yolov3-tiny (nc={nc}): expected "
                         f"{4 * n_floats + 16} or {4 * n_floats + 20} bytes for {n_floats} floats")
    buf = np.frombuffer(data[header:], dtype="<f4")

    ptr = 0

    def take(n):
        nonlocal ptr
        arr = torch.from_numpy(buf[ptr:ptr + n].astype(np.float32, copy=True))
        ptr += n
        return arr

    for prefix, kind, w_shape, n_out in plan:
        nw = int(np.prod(w_shape))
        if kind == "bn":
            beta, gamma, mean, var = take(n_out), take(n_out), take(n_out), take(n_out)
            sd[f"{prefix}.conv.weight"] = take(nw).reshape(w_shape)
            sd[f"{prefix}.bn.bias"] = beta
            sd[f"{prefix}.bn.weight"] = gamma
            sd[f"{prefix}.bn.running_mean"] = mean
            sd[f"{prefix}.bn.running_var"] = var
        else:
            sd[f"{prefix}.bias"] = take(n_out)
            sd[f"{prefix}.weight"] = take(nw).reshape(w_shape)
    assert ptr == buf.size  # 头部判别 + n_floats 精确时必然成立（防御性）
    save_file(sd, dst)
    return sd


def main():
    parser = argparse.ArgumentParser(description="official weights (.pt / darknet .weights) -> pure safetensors")
    parser.add_argument("--src", required=True, help="official weights path (.pt or darknet .weights)")
    parser.add_argument("--dst", required=True, help="output .safetensors path")
    parser.add_argument("--nc", type=int, default=80, help="class count (darknet .weights only)")
    args = parser.parse_args()

    if Path(args.src).suffix.lower() == ".weights":
        sd = _convert_darknet(args.src, args.dst, nc=args.nc)
    else:
        sd = _convert_pt(args.src, args.dst)

    keys = list(sd.keys())
    n_params = sum(v.numel() for v in sd.values())
    logger.info(f"saved {len(keys)} tensors, {n_params:,} params -> {args.dst}")
    logger.info(f"key first : {keys[0]}")
    logger.info(f"key middle: {keys[len(keys) // 2]}")
    logger.info(f"key last  : {keys[-1]}")


if __name__ == "__main__":
    main()
