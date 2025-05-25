# rainbow_yu exp7.multi_head_attention 🐋✨
# 多头注意力机制

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0

        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_weights = None

    def forward(self, x, mask=None):
        B, L, D = x.size()
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(B, L, self.num_heads, 3 * self.head_dim).transpose(1, 2)
        Q, K, V = torch.chunk(qkv, 3, dim=-1)

        scores = Q @ K.transpose(-2, -1) / (self.head_dim ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        self.attn_weights = attn.detach()
        out = attn @ V
        out = out.transpose(1, 2).reshape(B, L, D)
        return self.out_proj(out)
