"""scripts/convert_mobileclip_ts.py: mobileclip2_b.ts -> safetensors 转换契约

覆盖：ConstMap 按代码名到 state_dict 的映射布局（合成常量，不触网、不依赖大资产）、
头数断言、以及真实 .ts 的转换产物可被 model/text_encoder.py strict load
（资产闸门：weights/mobileclip2_b.ts 不存在则 skip）。
"""

import pytest
import torch
from safetensors.torch import save_file

from scripts.convert_mobileclip_ts import EXPECTED_SHAPES, build_state_dict

_BLOCK_SHAPES = (
    (512, 1536), (1536,),          # qkv
    (512, 512), (512,),            # out_proj
    (512,), (512,),                # ln_1
    (512, 2048), (2048,),          # fc1
    (2048, 512), (512,),           # fc2
    (512,), (512,),                # ln_2
)


def _synthetic_consts() -> dict[str, torch.Tensor]:
    shapes = {
        "c0": (49408, 512), "c1": (77, 512), "c2": (512,), "c3": (512,),
        "c7": (1, 1, 77, 77), "c150": (512, 512),
    }
    for i in range(12):
        names = (["c5", "c6", "c8", "c9", "c10", "c11", "c12", "c13", "c14",
                  "c15", "c16", "c17"] if i == 0
                 else [f"c{n}" for n in range(18 + 12 * (i - 1), 30 + 12 * (i - 1))])
        for n, s in zip(names, _BLOCK_SHAPES):
            shapes[n] = s
    consts = {k: torch.zeros(v, dtype=torch.float32) for k, v in shapes.items()}
    consts["c4"] = torch.zeros((), dtype=torch.int64) + 8
    return consts


def test_build_state_dict_layout():
    sd = build_state_dict(_synthetic_consts())
    assert len(sd) == 150  # 6 头部张量 + 12 块 × 12
    for key, shape in EXPECTED_SHAPES.items():
        assert tuple(sd[key].shape) == shape
    assert sd["blocks.0.attn.in_proj.weight"].shape == (512, 1536)
    assert sd["blocks.11.ln_2.bias"].shape == (512,)
    # 常量名（c4 头数 / c7 掩码）不进 state_dict
    assert not any(k.startswith("c") for k in sd)


def test_build_state_dict_rejects_wrong_heads():
    consts = _synthetic_consts()
    consts["c4"] = torch.zeros((), dtype=torch.int64) + 4
    with pytest.raises(ValueError, match="8 attention heads"):
        build_state_dict(consts)


def test_convert_real_ts_strict_load(tmp_path):
    from pathlib import Path

    ts = Path(__file__).resolve().parent.parent / "weights" / "mobileclip2_b.ts"
    if not ts.exists():
        pytest.skip("需要 weights/mobileclip2_b.ts "
                    "(python scripts/download_weights.py --model mobileclip2_b)")

    consts = torch.jit.load(ts, map_location="cpu").code_with_constants[1].const_mapping
    sd = build_state_dict(consts)
    assert len(sd) == 150
    out = tmp_path / "mobileclip2_b.safetensors"
    save_file(sd, out)

    from model.text_encoder import load_mobileclip_text

    encoder = load_mobileclip_text(out).eval()
    assert encoder is not None
