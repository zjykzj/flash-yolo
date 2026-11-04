"""OpenAI CLIP 字节级 BPE tokenizer（自写实现）

YOLOE 文本模式的官方 tokenizer = CLIP 的 BPE（ultralytics `clip.tokenize`）。
词表数据 `config/text/bpe_simple_vocab_16e6.txt.gz` 是 OpenAI CLIP 仓库的
MIT 数据文件（vendor 数据不抄代码）；本模块按公开的字节级 BPE 算法自写
（GPT-2/CLIP 谱系：字节 → unicode 字符映射、贪心合并、</w> 词尾标记）。

口径要点：
- 词表规模 49408：256 字节 token + 256 个带 </w> 的字节 token + 前 48894 条合并
  + SOS(49406) + EOS(49407)；EOS 是最大 token id——文本编码器的 EOT 位置池化
  （argmax 技巧）依赖这一点。
- 编码 = html.unescape×2 + strip → 空白折叠 → 小写 → GPT-2 式分词（缩略词/
  字母/数字/符号按类分组）→ 逐词 BPE(+</w>) → [SOS] + tokens + [EOS]。
- 与官方 fork 的偏差：官方 basic_clean 含 ftfy.fix_text（乱码编码修复，正常
  UTF-8 文本恒等）——不引入该依赖，乱码输入不承诺逐位一致。
- 截断（truncate=True，官方默认）：保留前 75 个 BPE token 再包 SOS/EOS；
  不满 77 用 0 补齐。
"""

from __future__ import annotations

import gzip
import html
import re
from functools import lru_cache
from pathlib import Path

__all__ = ["ClipTokenizer", "get_tokenizer", "tokenize"]

CONTEXT_LENGTH = 77
SOS_TOKEN = 49406
EOS_TOKEN = 49407
VOCAB_SIZE = 49408
_NUM_MERGES = VOCAB_SIZE - 2 - 256 - 256  # 48894（256 字节 + 256 带 </w> 字节 + 合并 + 2 特殊）

_VOCAB_PATH = Path(__file__).resolve().parent.parent / "config" / "text" / "bpe_simple_vocab_16e6.txt.gz"


def _bytes_to_unicode() -> dict[int, str]:
    """字节 → unicode 字符映射（GPT-2 谱系标准口径）：
    可打印 ASCII 与拉丁扩展保留原字符，其余字节顺延到补充码位。"""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


class ClipTokenizer:
    """CLIP BPE tokenizer（49408 词表，上下文 77）"""

    def __init__(self, vocab_path: str | Path = _VOCAB_PATH):
        self.byte_encoder = _bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        with gzip.open(vocab_path, "rt", encoding="utf-8") as f:
            lines = f.read().splitlines()
        # 文件首行是版本注释；前 N 条合并按行序 = 合并 rank
        merges = lines[1 : 1 + _NUM_MERGES]
        self.bpe_ranks = {tuple(m.split()): i for i, m in enumerate(merges)}
        # 词表（官方口径）：256 字节字符 + 256 个带 </w> 后缀的字节字符
        # （占 256..511）+ 合并 token + 特殊 token
        vocab = list(self.byte_encoder.values())
        vocab += [c + "</w>" for c in self.byte_encoder.values()]
        vocab += ["".join(m.split()) for m in merges]
        vocab += ["<|startoftext|>", "<|endoftext|>"]
        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.vocab_size = VOCAB_SIZE

    def _bpe(self, chars) -> list[str]:
        """贪心字节对合并（官方口径：每轮取 rank 最小的相邻对，**一次合并该对的
        所有出现**，左到右扫描；输入是字符序列，末字符已带 </w>）"""
        chars = list(chars)
        while len(chars) > 1:
            pairs = {(chars[i], chars[i + 1]) for i in range(len(chars) - 1)}
            best = min(pairs, key=lambda p: self.bpe_ranks.get(p, float("inf")))
            if best not in self.bpe_ranks:
                break
            a, b = best
            merged = []
            i = 0
            while i < len(chars):
                if chars[i] == a and i < len(chars) - 1 and chars[i + 1] == b:
                    merged.append(a + b)
                    i += 2
                else:
                    merged.append(chars[i])
                    i += 1
            chars = merged
        return chars

    _SPECIALS = ("<|startoftext|>", "<|endoftext|>")
    _CONTRACTIONS = ("'s", "'t", "'re", "'ve", "'m", "'ll", "'d")

    @staticmethod
    def _split(text: str) -> list[str]:
        """官方 fork 的 GPT-2 式分词（stdlib 模拟）

        官方 pattern = <|startoftext|>|<|endoftext|>|'s|'t|'re|'ve|'m|'ll|'d|
        [\\p{L}]+|[\\p{N}]|[^\\s\\p{L}\\p{N}]+（IGNORECASE，regex 模块）。
        本实现用 stdlib 等价替代：\\p{L}≈isalpha、\\p{N}≈isdigit；按 alternation
        顺序在每处取首个匹配，空白跳过（不产出 token）。
        """
        tokens = []
        i, n = 0, len(text)
        while i < n:
            rest = text[i : i + 15].lower()
            for special in ClipTokenizer._SPECIALS:
                if rest.startswith(special):
                    tokens.append(text[i : i + len(special)])
                    i += len(special)
                    break
            else:
                for cont in ClipTokenizer._CONTRACTIONS:
                    if rest.startswith(cont):
                        tokens.append(text[i : i + len(cont)])
                        i += len(cont)
                        break
                else:
                    ch = text[i]
                    if ch.isalpha():
                        j = i + 1
                        while j < n and text[j].isalpha():
                            j += 1
                        tokens.append(text[i:j])
                        i = j
                    elif ch.isdigit():
                        # 官方是 [\p{N}]（单字符，无 +）：数字逐个成词
                        tokens.append(ch)
                        i += 1
                    elif not ch.isspace():
                        j = i + 1
                        while j < n and not (text[j].isalpha() or text[j].isdigit() or text[j].isspace()):
                            j += 1
                        tokens.append(text[i:j])
                        i = j
                    else:
                        i += 1
        return tokens

    def _encode_text(self, text: str) -> list[int]:
        # 官方 basic_clean = ftfy.fix_text + html.unescape×2 + strip；ftfy 只修乱码
        # 编码，正常 UTF-8 文本恒等——为守住最小依赖不引入（乱码输入不承诺逐位）
        text = html.unescape(html.unescape(text)).strip()
        text = re.sub(r"\s+", " ", text).strip()
        text = text.lower()
        out: list[int] = []
        for word in self._split(text):
            # 官方 bpe 缓存预置两个特殊词：文本里出现即直接映射到特殊 token id
            if word in ("<|startoftext|>", "<|endoftext|>"):
                out.append(self.encoder[word])
                continue
            # 词 → UTF-8 字节 → 字节字符映射；GPT-2 口径 </w> 接在末字符上
            chars = [self.byte_encoder[b] for b in word.encode("utf-8")]
            chars[-1] += "</w>"
            out.extend(self.encoder[t] for t in self._bpe(chars))
        return out

    def encode(self, text: str) -> list[int]:
        """单条文本 → token id 列表（不含 SOS/EOS）"""
        return self._encode_text(text)

    def tokenize(self, texts: str | list[str], truncate: bool = True) -> list[list[int]]:
        """文本 → (B, 77) token 序列（SOS 开头、EOS 结尾、0 补齐）"""
        if isinstance(texts, str):
            texts = [texts]
        rows = []
        for text in texts:
            tokens = self._encode_text(text)
            if truncate:
                tokens = tokens[: CONTEXT_LENGTH - 2]
            row = [SOS_TOKEN] + tokens + [EOS_TOKEN]
            if len(row) > CONTEXT_LENGTH:
                raise RuntimeError(f"input is too long for context length {CONTEXT_LENGTH}")
            row += [0] * (CONTEXT_LENGTH - len(row))
            rows.append(row)
        return rows


@lru_cache(maxsize=None)
def get_tokenizer() -> ClipTokenizer:
    return ClipTokenizer()


def tokenize(texts: str | list[str], truncate: bool = True) -> "torch.Tensor":
    """官方 clip.tokenize 等价物：(B, 77) int64 张量"""
    import torch

    return torch.tensor(get_tokenizer().tokenize(texts, truncate=truncate), dtype=torch.int64)
