"""YOLOE 文本提示嵌入（推理引擎与训练 PE 共用的唯一实现）

链路（官方 get_text_pe 口径）：CLIP BPE tokenize -> MobileCLIP2-B 编码（L2 归一化）->
按批分块 -> (1, N, 512) -> head.get_tpe（reprta 残差 + L2 归一化）。

文本编码器权重 `mobileclip2_b.safetensors`（Apple mobileclip2_b.ts 的转换产物）不在
模型 ckpt 内：推理侧取权重同目录，训练侧经 `--text-encoder` 显式覆盖。
"""

from pathlib import Path

import torch

__all__ = ["TEXT_ENCODER_FILENAME", "resolve_text_encoder", "load_text_encoder", "encode_text_prompts"]

TEXT_ENCODER_FILENAME = "mobileclip2_b.safetensors"


def resolve_text_encoder(weights=None, explicit=None):
    """文本编码器路径：显式路径 > 模型权重同目录的 mobileclip2_b.safetensors

    Args:
        weights: 模型权重路径（.safetensors）；训练 PE 与推理都能提供
        explicit: 显式覆盖路径（CLI --text-encoder）
    """
    if explicit:
        p = Path(explicit)
    elif weights is not None:
        p = Path(weights).resolve().parent / TEXT_ENCODER_FILENAME
    else:
        raise ValueError("text encoder path cannot be resolved: pass an explicit path or a model "
                         f"weight path (for <weights dir>/{TEXT_ENCODER_FILENAME})")
    if not p.is_file():
        raise FileNotFoundError(
            f"text mode requires the MobileCLIP text encoder weights at {p} "
            "(pass --text-encoder to point at a different location)")
    return p


def load_text_encoder(path, device="cpu"):
    """加载 MobileCLIP2-B 文本编码器（eval 模式；与官方 ts 逐位对齐的纯推理件）"""
    from model.text_encoder import load_mobileclip_text

    return load_mobileclip_text(str(path)).to(device).eval()


def encode_text_prompts(head, texts, encoder, batch=80):
    """类名列表 -> head 所需的文本提示嵌入 (1, N, 512)（已过 get_tpe）

    逐 chunk detach：只切断 autograd 引用、不改数值；否则全部 chunk 的编码图会同时保留
    （1203 prompt 实测峰值 ~22G）。训练 PE 时 tpe 也是常量（不复用文本编码器梯度）。
    """
    from model.tokenizer import tokenize

    device = next(encoder.parameters()).device
    tokens = tokenize(list(texts)).to(device)
    emb = [encoder(chunk).detach() for chunk in tokens.split(batch)]
    emb = torch.cat(emb, dim=0).unsqueeze(0)  # (1, N, 512)
    return head.get_tpe(emb).detach()
