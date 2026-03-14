import torch
import torch.nn as nn

import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(seq_len, d_model) # (Seq_len, d_model)
        positions = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # (Seq_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # (d_model/2, )
        pe[:, 0::2] = torch.sin(positions * div_term) # (Seq_len, d_model)
        pe[:, 1::2] = torch.cos(positions * div_term) # (Seq_len, d_model)

        pe = pe.unsqueeze(0) # (1, seq_len, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.shape[1], :].requires_grad_(False)
        return self.dropout(x)


class PatchEmbedding(nn.Module):
    def __init__(self, image_size: int, in_channels: int, d_model: int, patch_size: int, positional_encoding: PositionalEncoding, dropout: float) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_channels=in_channels, out_channels=d_model, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        assert image_size % patch_size == 0, "image size must be divisible by patch size"
        num_patches = (image_size // patch_size) ** 2
        self.positional_encoding = PositionalEncoding(d_model, num_patches + 1, dropout)
    
    def forward(self, x: torch.Tensor):
        B = x.shape[0]
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.positional_encoding(x)
        return x
    

class LayerNormalizationBlock(nn.Module):
    def __init__(self, eps: float=10**-9) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, dropout:float, h: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h

        assert d_model % h == 0, "d_model is not divisible by h"
        self.d_k = d_model // h

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def selfattention(query, key, value, dropout: nn.Dropout):
        d_k = query.shape[-1]

        # (B, h, seq_len, d_k) -> (B, h, seq_len, seq_len)
        attention_score = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        attention_score = torch.softmax(attention_score, dim=-1)
        if dropout is not None:
            attention_score = dropout(attention_score)
        return (attention_score @ value), attention_score # (batch_size, h, seq_len, d_k)

    def forward(self, q, k, v):
        # (B, seq_len, d_model) -> (B, seq_len, d_model)
        query = self.w_q(q)
        key = self.w_q(k)
        value = self.w_q(v)

        # (B, seq_len, d_model) -> (B, seq_len, h, d_k) -> (B, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        x, self.attention_score = MultiHeadAttentionBlock.selfattention(query, key, value, self.dropout)
        
        # (B, h, seq_len, d_k) -> (B, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # (B, seq_len, d_model) -> (B, seq_len, d_model)
        return self.w_o(x)
    

class ResidualConnection(nn.Module):
    def __init__(self, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalizationBlock()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
    

class EncoderBlock(nn.Module):
    def __init__(self, self_attention: MultiHeadAttentionBlock, feed_forward: FeedForwardBlock, dropout: float):
        super().__init__()
        self.self_attention = self_attention
        self.feed_forward = feed_forward
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])
        self.norm = LayerNormalizationBlock()

    def forward(self, x):
        x = self.residual_connections[0](x, lambda x: self.self_attention(x, x, x))
        x = self.residual_connections[1](x, self.feed_forward)
        return self.norm(x)
    

class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalizationBlock()

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, class_size: int):
        super().__init__()
        self.proj = nn.Linear(d_model, class_size)

    def forward(self, x):
        return torch.log_softmax(self.proj(x[:, 0]), dim=-1)
    

class ViT(nn.Module):
    def __init__(self, patch_embedding: PatchEmbedding, encoder: Encoder, projection_layer: ProjectionLayer):
        super().__init__()
        self.patch_embedding = patch_embedding
        self.encoder = encoder
        self.projection_layer = projection_layer

    def encode(self, x):
        x = self.patch_embedding(x)
        x = self.encoder(x)
        return x

    def project(self, x):
        return self.projection_layer(x)
    

def build_vit(image_size: int, in_channels: int, patch_size: int, h: int=8, d_ff: int=2048, d_model: int=512, class_size: int=3, Nx: int=6, dropout: float=0.1):
    num_patches = (image_size // patch_size) ** 2
    positional_encoding = PositionalEncoding(d_model, num_patches, dropout)
    patch_embedding = PatchEmbedding(image_size, in_channels, d_model, patch_size, positional_encoding, dropout)

    encoder_blocks = []
    for _ in range(Nx):
        self_attention = MultiHeadAttentionBlock(d_model, dropout, h)
        feed_forward = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(self_attention, feed_forward, dropout)
        encoder_blocks.append(encoder_block)

    encoder = Encoder(nn.ModuleList(encoder_blocks))

    projection_layer = ProjectionLayer(d_model, class_size)

    vit = ViT(patch_embedding, encoder, projection_layer)

    for p in vit.parameters():
        if p.dim() > 1:
            nn.init.xavier_normal_(p)

    return vit
