"""MobileCLIP2-B 文本编码器（YOLOE 文本模式）

结构事实来自 Apple 的 TorchScript 资产 `mobileclip2_b.ts`（`TextModelWithNormalization`），
经私有工具转换为 safetensors 后 strict load。前向逐行对应 TorchScript 代码口径：
CLIP 文本编码器——token/位置嵌入 -> ln_pre -> 12 层 transformer（LN 在残差加完之后，
见 TextBlock）-> EOT 位置池化（argmax 技巧：EOS=49407 是最大 token id）->
512->512 投影 -> L2 归一化。

    token_embedding      Embedding(49408, 512)   CLIP 词表
    positional_embedding (77, 512)               上下文 77
    ln_pre               LayerNorm(512, 1e-5)
    blocks[0..11]        TextBlock：qkv 合投(512->1536, 有 bias) + 8 头 SDPA(因果掩码)
                         + out(512->512) + FFN(512->2048->512, exact gelu)
    text_projection      Linear(512, 512, bias=False)
    attn_mask            (1,1,77,77) 因果掩码（上三角 -inf），persistent buffer

与官方逐位对齐的注意点：LN eps=1e-5、gelu 精确式、qkv 单 Linear 合投、
池化取 argmax(tokens) 位置（截断无 EOS 时按最大 token id 的位置，官方同口径）。
文本编码器只跑 fp32（ts 前向把输入显式 cast 到 fp32）；导出时文本嵌入被
融合进检测头分类卷积（fuse），编码器本身不进 ONNX。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["MobileCLIPText", "TextBlock", "TextAttention", "TextMLP", "TextLinear",
           "load_mobileclip_text"]


class TextLinear(nn.Module):
    """x @ W 口径线性层：weight 存 (in, out)（官方 ts 的 matmul 布局）

    不用 nn.Linear——它的 addmm kernel 与 ts 的 matmul(bmm) 取整路径不同，
    12 层累计 ~1e-7 偏差；本模块与官方权重逐位对齐，weight 原样存官方布局。
    前向对权重做 detach：requires_grad=True 的输入会走 autograd 调度路径，
    matmul 结果与 plain tensor 路径差 ~5e-6（实测）——官方 ts 的常量无 grad，
    文本编码器在本仓库是纯推理件（M4 的文本嵌入也是离线编码），逐位对齐优先。
    """

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None

    def forward(self, x):
        out = torch.matmul(x, self.weight.detach())
        if self.bias is not None:
            out = torch.add(out, self.bias.detach(), alpha=1)
        return out


class TextAttention(nn.Module):
    """qkv 合并投影 + 多头发散注意力（MobileCLIP 文本块）

    in_proj 一次算出 q/k/v（官方权重即 512->1536 单投影），与逐头注意力
    共用因果掩码。数值口径 = TorchScript 里的 scaled_dot_product_attention
    （float 掩码直接加到注意力权重上）。
    """

    def __init__(self, dim=512, heads=8):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.in_proj = TextLinear(dim, 3 * dim, bias=True)
        self.out_proj = TextLinear(dim, dim, bias=True)

    def forward(self, x, mask):
        # 逐 op 对应 ts 图：seq-first (L,B,C) 布局 + 2D 输出投影——matmul/SDPA 的
        # kernel 路径由形状决定，batch-first 或 3D 输出投影会引入 ~1e-6 级取整差，
        # 被 LN 在近零方差位置放大到可见量级（实测 b0.ln_1 差 12）
        B, L, _ = x.shape
        x_t = x.transpose(1, 0)  # (L, B, C)
        qkv = self.in_proj(x_t)  # TextLinear（detach 口径）
        q, k, v = qkv.unflatten(-1, (3, self.dim)).permute(2, 0, 1, 3)
        q = q.reshape(L, B * self.heads, self.head_dim).transpose(0, 1)
        q = q.reshape(B, self.heads, L, self.head_dim)
        k = k.reshape(L, B * self.heads, self.head_dim).transpose(0, 1)
        k = k.reshape(B, self.heads, L, self.head_dim)
        v = v.reshape(L, B * self.heads, self.head_dim).transpose(0, 1)
        v = v.reshape(B, self.heads, L, self.head_dim)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        out = out.permute(2, 0, 1, 3).contiguous().reshape(B * L, self.dim)
        out = self.out_proj(out)  # TextLinear（detach 口径；2D matmul 对齐 ts）
        return out.reshape(L, B, self.dim).transpose(1, 0)


class TextMLP(nn.Module):
    """普通 FFN：512->2048 -> gelu -> 2048->512（x @ W 口径，见 TextLinear）"""

    def __init__(self, dim=512, hidden=2048):
        super().__init__()
        self.fc1 = TextLinear(dim, hidden, bias=True)
        self.fc2 = TextLinear(hidden, dim, bias=True)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class TextBlock(nn.Module):
    """MobileCLIP 文本块（ts 逐段对应）：注意力残差加 LN 之前的值

    前向有两个变量：x = 上一级的 pre-LN 值（残差基），x_ln = LN(x)（注意力输入）——
    注意力残差加 x 而不是 x_ln（官方口径，块 0 的残差基是 emb+pos，跳过 ln_pre）：

        x = x + attn(x_ln) -> x = x + mlp(ln_1(x)) -> 返回 (x, ln_2(x))

    ln_1 是注意力残差后的 LN（FFN 前）；ln_2 是块尾 LN——其输出作为下一块注意力的
    输入（与顶层 ln_pre 构成完整 LN 链）。块 0 的 (x, x_ln) = (emb+pos, ln_pre 输出)，
    块 11 的 ln_2 输出即池化输入。
    """

    def __init__(self, dim=512, heads=8, hidden=2048):
        super().__init__()
        self.attn = TextAttention(dim, heads)
        self.ln_1 = nn.LayerNorm(dim, eps=1e-5)
        self.mlp = TextMLP(dim, hidden)
        self.ln_2 = nn.LayerNorm(dim, eps=1e-5)

    def forward(self, x, x_ln, mask):
        x = x + self.attn(x_ln, mask)
        x = x + self.mlp(self.ln_1(x))
        return x, self.ln_2(x)


class MobileCLIPText(nn.Module):
    """MobileCLIP2-B 文本编码器（TextModelWithNormalization）

    forward(tokens) -> (B, 512) L2 归一化文本嵌入。
    tokens: (B, L) int64，L<=77，由 model/tokenizer.py 产出（含 SOS/EOS）。
    """

    def __init__(self):
        super().__init__()
        dim, heads, hidden, context = 512, 8, 2048, 77
        self.token_embedding = nn.Embedding(49408, dim)
        self.positional_embedding = nn.Parameter(torch.empty(context, dim))
        self.ln_pre = nn.LayerNorm(dim, eps=1e-5)
        self.blocks = nn.ModuleList(TextBlock(dim, heads, hidden) for _ in range(12))
        self.text_projection = TextLinear(dim, dim, bias=False)
        self.register_buffer(
            "attn_mask", torch.full((1, 1, context, context), float("-inf")).triu_(1)
        )

    def forward(self, tokens):
        L = tokens.shape[1]
        x = self.token_embedding(tokens) + self.positional_embedding[:L]
        x_ln = self.ln_pre(x)
        # 掩码裁到实际长度：官方 ts 恒吃 77 长输入（tokenizer 补齐），L=77 时逐位不变；
        # L<77 时官方 ts 直接报错，这里按因果语义裁剪（与 pos 嵌入的动态切片口径一致）
        mask = self.attn_mask[:, :, :L, :L].contiguous()
        for block in self.blocks:
            x, x_ln = block(x, x_ln, mask)
        x = x_ln
        # EOT 位置池化（官方 argmax 技巧；截断无 EOS 时同官方取最大 token id 位置）
        pooled = x[torch.arange(x.shape[0], device=x.device), tokens.argmax(-1)]
        out = self.text_projection(pooled)
        return out / out.norm(2, dim=-1, keepdim=True)


def load_mobileclip_text(path):
    """构建 MobileCLIPText 并 strict load 转换后的 safetensors"""
    from model.weights import load_weights

    model = MobileCLIPText()
    load_weights(model, path)
    return model
