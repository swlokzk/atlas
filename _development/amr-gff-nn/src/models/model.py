"""
model.py — GatedFusionFormer v3.0 完整模型定義

包含所有子模組與主模型，並在 forward() 中支援
active_modalities 參數以供消融實驗使用。

建構塊：
  LayerNorm         → channels-first 自定義 Layer Normalization
  Squeeze           → 維度壓縮工具模組
  ConvEncoder_IQ    → 深度可分離卷積局部特徵編碼器
  FusionTransformerBlock → Conv + Multi-Head Attention 混合 Transformer 塊
  GatingNetwork     → 多模態動態門控網絡
  GatedFusionFormer → 主模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from timm.layers import DropPath
except ImportError:
    from timm.models.layers import DropPath
from typing import List, Tuple


# ---------------------------------------------------------------------------
# 基礎模塊
# ---------------------------------------------------------------------------

class LayerNorm(nn.Module):
    """支援 channels-first 格式的 Layer Normalization。"""

    def __init__(self, normalized_shape: int, eps: float = 1e-6,
                 data_format: str = "channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias   = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape,
                                self.weight, self.bias, self.eps)
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight.view(1, -1, 1) * x + self.bias.view(1, -1, 1)


class Squeeze(nn.Module):
    """nn.Module 封裝，從張量中移除指定維度。"""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.squeeze(self.dim)


class ConvEncoder_IQ(nn.Module):
    """深度可分離卷積局部特徵編碼器。

    Args:
        dim            : 輸入/輸出通道數。
        hidden_dim_ratio: 擴張比率，默認 4。
        kernel_size    : 深度卷積核大小。
        drop_path      : DropPath 概率。
        drop_rate      : Dropout 概率。
    """

    def __init__(self, dim: int, hidden_dim_ratio: float = 4.,
                 kernel_size: int = 3, drop_path: float = 0.,
                 drop_rate: float = 0.1):
        super().__init__()
        hidden = int(hidden_dim_ratio * dim)
        self.dwconv   = nn.Conv1d(dim, dim, kernel_size,
                                  padding=kernel_size // 2, groups=dim)
        self.norm     = nn.BatchNorm1d(dim)
        self.pwconv1  = nn.Conv1d(dim, hidden, 1)
        self.act      = nn.GELU()
        self.pwconv2  = nn.Conv1d(hidden, dim, 1)
        self.dropout  = nn.Dropout(drop_rate)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(self.dwconv(x))
        x = self.act(self.pwconv1(x))
        x = self.dropout(x)
        x = self.dropout(self.pwconv2(x))
        return residual + self.drop_path(x)


class FusionTransformerBlock(nn.Module):
    """Conv + Multi-Head Self-Attention 混合 Transformer 塊。

    Args:
        dim       : 通道維度。
        num_heads : 注意力頭數。
        mlp_ratio : FFN 擴張比率。
        drop      : Dropout 概率。
        drop_path : DropPath 概率。
    """

    def __init__(self, dim: int, num_heads: int = 4,
                 mlp_ratio: float = 4., drop: float = 0.1,
                 drop_path: float = 0.):
        super().__init__()
        self.norm1        = LayerNorm(dim, data_format="channels_first")
        self.conv_encoder = ConvEncoder_IQ(dim=dim, drop_path=drop_path,
                                           drop_rate=drop)
        self.norm2 = LayerNorm(dim, data_format="channels_first")
        self.attn  = nn.MultiheadAttention(dim, num_heads,
                                           dropout=drop, batch_first=True)
        self.norm3 = LayerNorm(dim, data_format="channels_first")
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Conv1d(dim, hidden, 1), nn.GELU(), nn.Dropout(drop),
            nn.Conv1d(hidden, dim, 1), nn.Dropout(drop),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.conv_encoder(self.norm1(x)))
        x_t = self.norm2(x).permute(0, 2, 1)
        attn_out, _ = self.attn(x_t, x_t, x_t)
        x = x + self.drop_path(attn_out.permute(0, 2, 1))
        x = x + self.drop_path(self.mlp(self.norm3(x)))
        return x


class GatingNetwork(nn.Module):
    """多模態動態門控網絡。

    計算 IQ、STFT、S-TD 三種模態的融合權重 (Softmax)。

    Args:
        input_dim  : 特徵維度。
        hidden_dim : MLP 隱藏層維度。
    """

    def __init__(self, input_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.mlp  = nn.Sequential(
            nn.Linear(input_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
            nn.Softmax(dim=1),
        )

    def forward(self, feat_iq: torch.Tensor,
                feat_stft: torch.Tensor,
                feat_std: torch.Tensor) -> torch.Tensor:
        p = lambda f: self.pool(f).squeeze(-1)
        return self.mlp(torch.cat([p(feat_iq), p(feat_stft), p(feat_std)], dim=1))


# ---------------------------------------------------------------------------
# 主模型
# ---------------------------------------------------------------------------

class GatedFusionFormer(nn.Module):
    """GatedFusionFormer v3.0 — 多模態自動調製識別主模型。

    Inputs:
        x_iq   : (B, 2, T)      原始 IQ 信號
        x_stft : (B, 1, 32, T)  STFT 時頻圖
        x_std  : (B, 2, T)      S-TD 二階統計特徵

    Returns:
        logits  : (B, num_classes)
        weights : (B, 3)  三模態動態融合權重

    Args:
        embed_dim       : 內部隱藏維度。
        num_classes     : 分類類別數。
        stft_time_steps : 時間維度對齊長度。
        depth           : Transformer 塊堆疊深度。
        num_heads       : 注意力頭數。
        active_modalities (forward only): 用於消融實驗，指定啟用的模態列表，
                          元素為 'iq'、'stft'、'std'。
    """

    def __init__(self, embed_dim: int = 96, num_classes: int = 11,
                 stft_time_steps: int = 128, depth: int = 4,
                 num_heads: int = 4):
        super().__init__()
        self.embed_iq = nn.Conv1d(2, embed_dim, kernel_size=7,
                                  padding=3, bias=False)
        self.embed_stft = nn.Sequential(
            nn.Conv2d(1, embed_dim // 2, 3, padding=1),
            nn.BatchNorm2d(embed_dim // 2), nn.ReLU(),
            nn.Conv2d(embed_dim // 2, embed_dim, 3, padding=1),
            nn.BatchNorm2d(embed_dim), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, stft_time_steps)),
            Squeeze(2),
        )
        self.embed_std     = nn.Conv1d(2, embed_dim, kernel_size=7,
                                       padding=3, bias=False)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(stft_time_steps)
        self.gating_network = GatingNetwork(input_dim=embed_dim)
        self.backbone = nn.Sequential(*[
            FusionTransformerBlock(dim=embed_dim, num_heads=num_heads)
            for _ in range(depth)
        ])
        self.norm    = LayerNorm(embed_dim, eps=1e-6, data_format="channels_first")
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head    = nn.Linear(embed_dim, num_classes)

    def forward(
        self,
        x_iq:   torch.Tensor,
        x_stft: torch.Tensor,
        x_std:  torch.Tensor,
        active_modalities: List[str] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if active_modalities is None:
            active_modalities = ['iq', 'stft', 'std']
        feat_iq   = self.adaptive_pool(self.embed_iq(x_iq))
        feat_stft = self.embed_stft(x_stft)
        feat_std  = self.adaptive_pool(self.embed_std(x_std))

        # 消融支援：將不活躍模態的特徵置零
        if 'iq'   not in active_modalities:
            feat_iq   = torch.zeros_like(feat_iq)
        if 'stft' not in active_modalities:
            feat_stft = torch.zeros_like(feat_stft)
        if 'std'  not in active_modalities:
            feat_std  = torch.zeros_like(feat_std)

        weights = self.gating_network(feat_iq, feat_stft, feat_std)
        w_iq, w_stft, w_std = weights[:, 0:1], weights[:, 1:2], weights[:, 2:3]

        fused = (feat_iq   * w_iq.unsqueeze(-1) +
                 feat_stft * w_stft.unsqueeze(-1) +
                 feat_std  * w_std.unsqueeze(-1))

        x = self.avgpool(self.norm(self.backbone(fused))).squeeze(-1)
        return self.head(x), weights
