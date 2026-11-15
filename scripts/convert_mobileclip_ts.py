#!/usr/bin/env python3
"""convert Apple mobileclip2_b.ts (TorchScript) -> safetensors for model/text_encoder.py

来源资产：`mobileclip2_b.ts`（~254 MB）经 `scripts/download_weights.py --model mobileclip2_b`
从 ultralytics assets release 下载（Apple MobileCLIP2-B 文本编码器的 TorchScript 资产）。
**许可：该权重为 Apple Machine Learning Research Model (AMLR) 许可——仅限研究用途、
非商用**；本仓库不随 Release 分发，由用户自行下载并接受条款。本转换器为自写代码。

提取口径：torch.jit.load(map_location="cpu") + `code_with_constants` 的 ConstMap
按代码名取值（c0..c150）。注意：ts 归档里 constants.pkl 的元组顺序与代码名
存在同形状组内置换——绝不能按归档位置硬编号，必须按名字取。

结构事实（2026-11-04，从 code_with_constants 代码还原）：
    c0        token_embedding.weight   (49408, 512)   CLIP 词表
    c1        positional_embedding     (77, 512)      上下文 77
    c2, c3    ln_pre.weight/bias       (512,)         eps=1e-5
    c4        num_heads                () int64 == 8  （断言，不写入）
    c7        attn_mask                (1,1,77,77)    因果掩码：上三角 -inf
    12 块（LN 在残差加完之后）：
      block 0 = c5,c6,(c7),c8..c17；block i>=1 = c(18+12(i-1))..c(29+12(i-1))
      每块：qkv_w qkv_b [mask] out_w out_b ln_1_w ln_1_b fc1_w fc1_b fc2_w fc2_b ln_2_w ln_2_b
      qkv 合投 (512,1536)、out (512,512)、fc1 (512,2048)、fc2 (2048,512)
      ln_1 = 注意力残差后的 LN（FFN 前）；ln_2 = 块尾 LN（下一块注意力的输入）
    c150       text_projection.weight   (512, 512)    无 bias
  前向 = CLIP 文本编码器口径（EOT 位置池化 + 投影 + L2 归一化）。

线性层布局：ts 的 matmul 是 x @ W 口径（权重存 (in, out)）；model/text_encoder.py 的
TextLinear 同口径（torch.matmul 与 ts 同 kernel 路径，逐位对齐）——权重**原样**
存储，不做任何转置。nn.Linear 的 addmm 与 matmul 取整路径不同，12 层累计
~1e-7 偏差，曾实测踩过。

用法:
  python scripts/convert_mobileclip_ts.py \
      --input weights/mobileclip2_b.ts --output weights/mobileclip2_b.safetensors
"""

from __future__ import annotations

import argparse

import torch
from safetensors.torch import save_file

EXPECTED_SHAPES = {
    "token_embedding.weight": (49408, 512),
    "positional_embedding": (77, 512),
    "text_projection.weight": (512, 512),
    "attn_mask": (1, 1, 77, 77),
}


def build_state_dict(consts: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    heads = int(consts["c4"].item())
    if heads != 8:
        raise ValueError(f"expected 8 attention heads, got {heads}")

    sd: dict[str, torch.Tensor] = {}
    sd["token_embedding.weight"] = consts["c0"]
    sd["positional_embedding"] = consts["c1"]
    sd["ln_pre.weight"] = consts["c2"]
    sd["ln_pre.bias"] = consts["c3"]
    sd["attn_mask"] = consts["c7"]
    sd["text_projection.weight"] = consts["c150"]

    block_keys = (
        "attn.in_proj.weight", "attn.in_proj.bias",
        "attn.out_proj.weight", "attn.out_proj.bias",
        "ln_1.weight", "ln_1.bias",
        "mlp.fc1.weight", "mlp.fc1.bias",
        "mlp.fc2.weight", "mlp.fc2.bias",
        "ln_2.weight", "ln_2.bias",
    )
    for i in range(12):
        if i == 0:
            names = ["c5", "c6", "c8", "c9", "c10", "c11", "c12", "c13", "c14", "c15", "c16", "c17"]
        else:
            start = 18 + 12 * (i - 1)
            names = [f"c{n}" for n in range(start, start + 12)]
        for suffix, name in zip(block_keys, names):
            sd[f"blocks.{i}.{suffix}"] = consts[name]

    for key, shape in EXPECTED_SHAPES.items():
        if tuple(sd[key].shape) != shape:
            raise ValueError(f"{key}: shape {tuple(sd[key].shape)} != {shape}")
    return sd


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--input", required=True, help="path to mobileclip2_b.ts")
    ap.add_argument("--output", required=True, help="output .safetensors path")
    args = ap.parse_args()

    # map_location 必须显式 cpu：ts 被 trace 在 cuda:0 上，常量默认留在原设备
    model = torch.jit.load(args.input, map_location="cpu")
    consts = model.code_with_constants[1].const_mapping
    sd = build_state_dict(consts)
    n_params = sum(t.numel() for t in sd.values())
    save_file(sd, args.output)
    print(f"wrote {args.output}: {len(sd)} tensors, {n_params:,} params "
          f"({n_params * 4 / 1e6:.1f} MiB fp32)")


if __name__ == "__main__":
    main()
