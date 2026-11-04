"""MobileCLIP2-B 文本编码器 + CLIP BPE tokenizer 验收（M2 文本模式）

权重依赖（dev-time，工具在私有仓库 tools/convert_mobileclip_ts.py）：
    python tools/convert_mobileclip_ts.py \
        --input weights/mobileclip2_b.ts --output weights/mobileclip2_b.safetensors

结构事实（ts 实测）：12 层 512-d transformer（8 头 × 64 维）、LN 在残差加完之后
（注意力残差加 pre-LN 值）、线性层 x @ W 口径（weight 存 (in, out)）、
EOT 位置池化（argmax 技巧，EOS=49407 是最大 token id）、77 上下文因果掩码。
tokenizer 与官方 fork 逐位一致（除 ftfy 乱码修复/全角转换，正常 UTF-8 文本恒等）。
"""

from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
TS_PATH = ROOT / "weights" / "mobileclip2_b.ts"
SAFE_PATH = ROOT / "weights" / "mobileclip2_b.safetensors"

N_PARAMS = 63_428_096  # parameters() 口径；attn_mask buffer 另计 5,929（转换产物共 63,434,025）


def _encoder():
    from model.text_encoder import MobileCLIPText

    return MobileCLIPText()


# ---------- 结构（无权重依赖） ----------

def test_param_count():
    """结构硬指标：63,434,025 参数（与转换产物张量数对账）"""
    assert sum(p.numel() for p in _encoder().parameters()) == N_PARAMS


def test_text_linear_layout():
    """TextLinear 权重存 (in, out)（官方 x @ W 口径；nn.Linear 是 (out, in)）"""
    from model.text_encoder import TextLinear

    lin = TextLinear(512, 1536)
    assert tuple(lin.weight.shape) == (512, 1536)


def test_block_structure_and_mask():
    """结构硬指标：12 块、8 头 64 维、因果掩码（上三角 -inf）"""
    m = _encoder()
    assert len(m.blocks) == 12
    attn = m.blocks[0].attn
    assert attn.heads == 8 and attn.head_dim == 64
    assert tuple(m.attn_mask.shape) == (1, 1, 77, 77)
    assert bool((m.attn_mask[0, 0] == torch.full((77, 77), float("-inf")).triu(1)).all())


def test_forward_shapes_and_norm():
    """L<77 可跑（官方 ts 只吃 77，本实现按因果语义裁掩码）、输出单位范数"""
    torch.manual_seed(0)
    m = _encoder()
    for p in m.parameters():
        torch.nn.init.normal_(p)
    m.eval()
    for L in (77, 10, 3):
        tokens = torch.randint(0, 49407, (2, L))
        tokens[:, -1] = 49407
        with torch.no_grad():
            out = m(tokens)
        assert tuple(out.shape) == (2, 512)
        assert bool(torch.isfinite(out).all())
        assert bool((out.norm(2, dim=-1) - 1).abs().max() < 1e-5)


# ---------- tokenizer ----------

GOLDEN = {
    # 与官方 fork 对拍固定的黄金序列（49406=SOS、49407=EOS）
    "a photo of a cat": [49406, 320, 1125, 539, 320, 2368, 49407],
    "person": [49406, 2533, 49407],
    "dinning table": [49406, 1462, 762, 2175, 49407],  # 官方词汇自己的拼写
    "hello world 123": [49406, 3306, 1002, 272, 273, 274, 49407],  # 数字逐个成词
}


@pytest.mark.parametrize("text,expected", list(GOLDEN.items()))
def test_tokenize_golden(text, expected):
    from model.tokenizer import tokenize

    row = tokenize(text)[0]
    assert row[row != 0].tolist() == expected


def test_tokenize_padding_and_truncation():
    from model.tokenizer import tokenize

    row = tokenize("a photo of a cat")[0]
    assert tuple(row.shape) == (77,)
    assert row[-1] == 0
    long_row = tokenize("x " * 300)[0]  # 截断口径：保留前 75 个 BPE token
    nz = long_row[long_row != 0].tolist()
    assert nz[0] == 49406 and nz[-1] == 49407 and len(nz) <= 77


def test_tokenize_specials_and_contractions():
    """文本里的特殊词直接映射到特殊 token id（官方 bpe 缓存预置口径）"""
    from model.tokenizer import tokenize

    row = tokenize("<|startoftext|> weird")[0]
    nz = row[row != 0].tolist()
    assert nz[0] == 49406 and nz[1] == 49406  # SOS + 内容里的特殊词
    # 缩略词独立成词（GPT-2 式分词）：don + 't
    assert tokenize("don't")[0][1:3].tolist() == [847, 713]


def test_tokenizer_vocab_invariants():
    from model.tokenizer import get_tokenizer

    tok = get_tokenizer()
    assert tok.vocab_size == 49408
    assert tok.encoder["<|startoftext|>"] == 49406
    assert tok.encoder["<|endoftext|>"] == 49407  # 最大 id——EOT 池化依赖
    assert tok.encoder["a</w>"] == 320  # 256 字节 + 256 带 </w> 字节中的 'a'


# ---------- 官方逐位对齐（需要权重） ----------

def test_strict_load():
    if not SAFE_PATH.exists():
        pytest.skip("需要转换产物 weights/mobileclip2_b.safetensors")
    from model.text_encoder import MobileCLIPText
    from model.weights import load_weights

    m = MobileCLIPText()
    missing, unexpected = load_weights(m, SAFE_PATH)
    assert not missing and not unexpected


def test_bit_parity_vs_official_ts():
    """与官方 mobileclip2_b.ts 逐位一致（CPU；L=77 是官方输入契约）"""
    if not (TS_PATH.exists() and SAFE_PATH.exists()):
        pytest.skip("需要官方 .ts 与转换产物")
    from model.text_encoder import load_mobileclip_text

    torch.manual_seed(0)
    mine = load_mobileclip_text(SAFE_PATH).eval()
    ts = torch.jit.load(TS_PATH, map_location="cpu")
    for batch in (3, 1):
        tokens = torch.randint(0, 49407, (batch, 77))
        tokens[:, -1] = 49407
        with torch.no_grad():
            out_mine, out_ts = mine(tokens), ts(tokens)
        assert bool((out_mine == out_ts).all()), "与官方 ts 输出必须逐位一致"
