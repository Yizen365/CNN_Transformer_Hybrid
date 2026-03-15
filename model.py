import torch
import torch.nn as nn

import math


class Stem(nn.Module):
    def __init__(self, in_channels: int=3, out_channels: int=64):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
    
    def forward(self, x):
        return self.stem(x)
    

class Conv(nn.Module):
    def __init__(self, in_channels: int, expansion: int=4, stride: int=1):
        super().__init__()
        hidden_dim = in_channels * expansion
        self.expand = nn.Conv2d(in_channels, out_channels=hidden_dim, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.d_conv = nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim)
        self.bn2 = nn.BatchNorm2d(hidden_dim)
        self.proj = nn.Conv2d(in_channels=hidden_dim, out_channels=in_channels, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(in_channels)
        self.act = nn.GELU()
        self.use_residual = stride == 1

    def forward(self, x):
        residual = x
        x = self.act(self.bn1(self.expand(x)))
        x = self.act(self.bn2(self.d_conv(x)))
        x = self.bn3(self.proj(x))
        if self.use_residual:
            return x + residual
        return x
    

class ConvStage(nn.Module):
    def __init__(self, in_channels: int, num_blocks: int=3):
        super().__init__()
        layers = []
        layers.append(Conv(in_channels, stride=2))
        for _ in range(num_blocks - 1):
            layers.append(Conv(in_channels))
        self.blocks = nn.Sequential(*layers)

    def forward(self, x):
        return self.blocks(x)
    

class HybridConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = Stem(3, 64)
        self.stage1 = ConvStage(64, num_blocks=2)
        self.stage2 = ConvStage(64, num_blocks=2)

    def forward(self, x):
        return self.stage2(self.stage1(self.stem(x))) # (B, 64, 28, 28)


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
    def __init__(self, image_size: int, in_channels: int, d_model: int, patch_size: int, dropout: float) -> None:
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
    

class Hybrid_ViT(nn.Module):
    def __init__(self, conv_layer: HybridConv, patch_embedding: PatchEmbedding, encoder: Encoder, projection_layer: ProjectionLayer):
        super().__init__()
        self.cnn = conv_layer
        self.patch_embedding = patch_embedding
        self.transformer = encoder
        self.projection_layer = projection_layer

    def encode(self, x):
        x = self.cnn(x)
        x = self.patch_embedding(x)
        x = self.transformer(x)
        return x

    def project(self, x):
        return self.projection_layer(x)
    

def build_hybrid_vit(config, dropout: float=0.1):

    conv_layer = HybridConv()

    num_patches = (config['image_size'] // config['patch_size']) ** 2
    positional_encoding = PositionalEncoding(config['d_model'], num_patches, dropout)
    patch_embedding = PatchEmbedding(config['image_size'], config['trans_in_channels'], config['d_model'], config['patch_size'], dropout)

    encoder_blocks = []
    for _ in range(config['layers']):
        self_attention = MultiHeadAttentionBlock(config['d_model'], dropout, config['heads'])
        feed_forward = FeedForwardBlock(config['d_model'], config['mlp_dim'], dropout)
        encoder_block = EncoderBlock(self_attention, feed_forward, dropout)
        encoder_blocks.append(encoder_block)

    encoder = Encoder(nn.ModuleList(encoder_blocks))

    projection_layer = ProjectionLayer(config['d_model'], config['class_size'])

    vit = Hybrid_ViT(conv_layer, patch_embedding, encoder, projection_layer)

    for m in vit.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)

    return vit
