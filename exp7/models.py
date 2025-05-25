# rainbow_yu exp7.models 🐋✨
# 模型

from multi_head_attention import MultiHeadSelfAttention
import torch.nn as nn
import torch

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pos = torch.arange(0, max_len).unsqueeze(1)
        i = torch.arange(0, d_model, 2)
        angle_rates = 1 / torch.pow(10000, (i.float() / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * angle_rates)
        pe[:, 1::2] = torch.cos(pos * angle_rates)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)

class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_classes=2, pad_idx=0):
        super().__init__()
        self.pad_idx = pad_idx
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim)
        self.attention = MultiHeadSelfAttention(embed_dim, num_heads)
        self.fc = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x):
        mask = (x != self.pad_idx).unsqueeze(1).unsqueeze(2)
        emb = self.embedding(x)
        emb = self.pos_encoder(emb)
        attn_out = self.attention(emb, mask)
        pooled = attn_out.mean(dim=1)
        return self.fc(pooled)

class MHAClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_classes=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.linear = nn.Linear(embed_dim, num_classes)
        self.attn_weights = None  # 用于保存注意力权重

    def forward(self, x):
        x = self.embedding(x)  # [B, T, E]
        attn_output, attn_weights = self.attn(x, x, x)
        self.attn_weights = attn_weights  # [B, T, T]
        pooled = attn_output.mean(dim=1)
        return self.linear(pooled)
