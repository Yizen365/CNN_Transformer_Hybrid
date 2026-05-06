import torch
import torch.nn as nn

import math


class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super().__init__()
        hidden_dim = int(in_channels * expand_ratio)
        self.use_residual = (stride == 1 and in_channels == out_channels)

        layers = []

        # Expansion (1x1 conv)
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
            ])

        # Depthwise
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 3,
                      stride=stride,
                      padding=1,
                      groups=hidden_dim,
                      bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
        ])

        # Projection (Linear bottleneck)
        layers.extend([
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self):
        super().__init__()

        # (expand_ratio, channels, repeats, stride)
        cfg = [
            (1, 16, 1, 1),
            (6, 24, 2, 2),
            (6, 32, 3, 2),
            (6, 64, 4, 2),
            (6, 96, 3, 1),
            (6, 160, 3, 2),
            (6, 320, 1, 1),
        ]

        layers = []

        # Initial conv
        input_channel = 32
        layers.extend([
            nn.Conv2d(3, input_channel, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.ReLU6(inplace=True),
        ])

        # Inverted residual blocks
        for t, c, n, s in cfg:
            for i in range(n):
                stride = s if i == 0 else 1
                layers.append(
                    InvertedResidual(input_channel, c, stride, expand_ratio=t)
                )
                input_channel = c

        # Final layer
        layers.extend([
            nn.Conv2d(input_channel, 1280, 1, bias=False),
            nn.BatchNorm2d(1280),
            nn.ReLU6(inplace=True),
        ])

        self.features = nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        return x


class CNN_Backbone(nn.Module):
    def __init__(self, d_model: int=512):
        super().__init__()
        self.cnn = MobileNetV2()
        self.proj = nn.Linear(1280, d_model)
    
    def forward(self, x: torch.Tensor):
        x = self.cnn(x) # output is (B, 1280, 7, 7)
        x = x.flatten(2).transpose(1, 2) # (B, 1280, 7, 7) -> (B, 1280, 49) -> (B, 49, 1280)
        x = self.proj(x) # (B, 49, 1280) -> (B, 49, 512)
        return x


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
    def __init__(self, d_model: int=512, eps: float=1e-9) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        return self.alpha * (x - mean) / torch.sqrt(var + self.eps) + self.bias


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
        key = self.w_k(k)
        value = self.w_v(v)

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
        self.norm = LayerNormalizationBlock(d_model=512)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
    

class EncoderBlock(nn.Module):
    def __init__(self, self_attention: MultiHeadAttentionBlock, feed_forward: FeedForwardBlock, dropout: float):
        super().__init__()
        self.self_attention = self_attention
        self.feed_forward = feed_forward
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])
        
    def forward(self, x):
        x = self.residual_connections[0](x, lambda x: self.self_attention(x, x, x))
        x = self.residual_connections[1](x, self.feed_forward)
        return x
    

class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalizationBlock(d_model=512)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, class_size: int):
        super().__init__()
        self.proj = nn.Linear(d_model, class_size)

    def forward(self, x):
        return self.proj(x)
    

class CrossAttentionFusion(nn.Module):
    def __init__(self, d_model: int, dropout: float, h: int):
        super().__init__()
        self.cross_attention = MultiHeadAttentionBlock(d_model, dropout, h)
        self.w_o = nn.Linear(out_features=d_model, in_features=2*d_model)
        self.norm = LayerNormalizationBlock(d_model=d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        normed_query = self.norm(query)
        normed_key = self.norm(key)
        normed_value = self.norm(value)
        attention = self.cross_attention(normed_query, normed_key, normed_value)
        fused_attention = torch.cat([normed_query, attention], dim=-1)
        fused_attention = self.dropout(self.w_o(fused_attention))
        fused_attention = fused_attention + query
        return fused_attention

    

class Hybrid_CNN_ViT(nn.Module):
    def __init__(self, conv_layer: CNN_Backbone, patch_embedding: PatchEmbedding, encoder: Encoder, cross_attention_fuse: CrossAttentionFusion, projection_layer: ProjectionLayer):
        super().__init__()
        self.cnn_backbone = conv_layer
        self.patch_embedding = patch_embedding
        self.encoder = encoder
        self.cross_attention_fuse = cross_attention_fuse
        self.projection_layer = projection_layer

    def convolution(self, x):
        x = self.cnn_backbone(x)
        return x

    def encode(self, x):
        x = self.patch_embedding(x)
        x = self.encoder(x)
        return x
    
    def cross_attention_fusion(self, query, key, value):
        return self.cross_attention_fuse(query, key, value)

    def project(self, x):
        return self.projection_layer(x)
    
    def forward(self, x):
        x1 = self.convolution(x)
        x2 = self.encode(x)
        cross_attention_fused = self.cross_attention_fusion(x2, x1, x1)
        cls_token = cross_attention_fused[:, 0]
        return self.project(cls_token)
    

def init_vit_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


def build_hybrid(config, dropout: float=0.2):

    conv_layer = CNN_Backbone()

    patch_embedding = PatchEmbedding(config['image_size'], config['in_channels'], config['d_model'], config['patch_size'], dropout)

    encoder_blocks = []
    for _ in range(config['layers']):
        self_attention = MultiHeadAttentionBlock(config['d_model'], dropout, config['heads'])
        feed_forward = FeedForwardBlock(config['d_model'], config['mlp_dim'], dropout)
        encoder_block = EncoderBlock(self_attention, feed_forward, dropout)
        encoder_blocks.append(encoder_block)

    encoder = Encoder(nn.ModuleList(encoder_blocks))

    cross_attention_fuse = CrossAttentionFusion(config['d_model'], dropout, config['heads'])

    projection_layer = ProjectionLayer(config['d_model'], config['class_size'])

    hybrid = Hybrid_CNN_ViT(conv_layer, patch_embedding, encoder, cross_attention_fuse, projection_layer)

    init_vit_weights(hybrid.patch_embedding)
    init_vit_weights(hybrid.encoder)
    init_vit_weights(hybrid.cross_attention_fuse)
    init_vit_weights(hybrid.projection_layer)


    return hybrid