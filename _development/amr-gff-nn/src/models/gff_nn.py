import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class ConvFFN(nn.Module) : 
    """卷積版前饋網絡 (Convolutional Feed-Forward Network)。
    
    支援多種現代激活函數門控機制 (GELU, GEGLU, SwiGLU)，適用於處理 1D 序列信號。
    相較於傳統 FFN，門控機制能更好地捕捉調制信號中的非線性特徵。

    Args:
        dim (int): 輸入與輸出的通道維度。
        hidden_dim (int): 隱藏層維度，通常為 dim * 4。
        drop (float): Dropout 概率。
        activation (str): 激活函數類型，可選 "gelu", "geglu", "swiglu"。

    Shapes:
        - Input: (B, C, T) 其中 B 為 Batch size, C 為通道數, T 為時間步長。
        - Output: (B, C, T)
    """
    def __init__(self, dim: int, hidden_dim: int, drop: float = 0.1, activation: str = "geglu"):
        super().__init__()
        act = activation.lower()
        self.act_kind = act
        if act == "gelu":
            self.fc1 = nn.Conv1d(dim, hidden_dim, 1)
        elif act in ("geglu", "swiglu"):
            self.fc1 = nn.Conv1d(dim, hidden_dim * 2, 1)
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        self.fc2 = nn.Conv1d(hidden_dim, dim, 1)
        self.dropout = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """執行前向傳播。"""
        if self.act_kind == "gelu":
            x = self.fc1(x)
            x = F.gelu(x)
        else:
            x_proj = self.fc1(x)
            a, gate = x_proj.chunk(2, dim=1)
            if self.act_kind == "geglu":
                gate = F.gelu(gate)
            else:  # swiglu
                gate = F.silu(gate)
            x = a * gate
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class GatingNetwork(nn.Module):
    """多模態動態門控網絡 (Dynamic Gating Network)。
    
    用於計算 IQ、STFT 以及統計特徵 (STD) 三者的融合權重。
    通過 Softmax 溫度調節與底噪 (Epsilon) 確保融合過程的平滑性與模態多樣性。

    Args:
        input_dim (int): 特徵向量的維度。
        hidden_dim (int): 內部 MLP 的隱藏層維度。
        temperature (float): Softmax 溫度係數，越高權重越均勻，越低權重越集中。
        eps (float): 歸一化底噪，防止某個分支權重完全歸零。

    Shapes:
        - Input (fiq, fstft, fstd): (B, C, T)
        - Output (weights): (B, 3) 代表三種模態的動態權重。
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64, temperature: float = 1.5, eps: float = 0.02):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.bn_iq  = nn.LayerNorm(input_dim)
        self.bn_st  = nn.LayerNorm(input_dim)
        self.bn_std = nn.LayerNorm(input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim * 3, hidden_dim), 
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)
        )
        self.temperature = temperature
        self.eps = eps
        
    def forward(self, fiq: torch.Tensor, fstft: torch.Tensor, fstd: torch.Tensor) -> torch.Tensor:
        """計算各個模態的動態融合權重。"""
        piq  = self.bn_iq(self.pool(fiq).squeeze(-1))
        pst  = self.bn_st(self.pool(fstft).squeeze(-1))
        pstd = self.bn_std(self.pool(fstd).squeeze(-1))
        logits = self.mlp(torch.cat([piq, pst, pstd], dim=1))
        w = torch.softmax(logits / self.temperature, dim=1)
        w = (w + self.eps) / (w + self.eps).sum(dim=1, keepdim=True)
        return w


class StftEncoder2D(nn.Module):
    """2D 時頻編碼器 (STFT Encoder)。
    
    利用 2D 卷積提取時頻圖特徵，並通過頻率注意力機制 (Frequency Attention) 
    將頻率維度聚合，輸出與 1D 信號對齊的時間特徵序列。

    Args:
        in_ch (int): 輸入通道數，通常為 1 (單色時頻圖)。
        embed_dim (int): 嵌入特徵維度。
        drop (float): Dropout 概率。

    Shapes:
        - Input: (B, 1, F, T) F 為頻率點數, T 為時間步長。
        - Output: (B, embed_dim, T)
    """
    def __init__(self, in_ch: int = 1, embed_dim: int = 96, drop: float = 0.1):
        super().__init__()
        self.feat2d = nn.Sequential(
            nn.Conv2d(in_ch, embed_dim // 2, 3, padding=1), 
            nn.BatchNorm2d(embed_dim // 2), 
            nn.ReLU(),
            nn.Conv2d(embed_dim // 2, embed_dim, 3, padding=1), 
            nn.BatchNorm2d(embed_dim), 
            nn.ReLU(),
        )
        self.freq_att = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 1), 
            nn.ReLU(),
            nn.Conv2d(embed_dim, 1, 1)
        )
        self.dropout = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """提取 2D 特徵並執行頻域加權聚合。"""
        x = self.feat2d(x)
        w = self.freq_att(x)
        w = torch.softmax(w, dim=2) 
        x = (x * w).sum(dim=2)
        return self.dropout(x)


class GatedFusionFormer(nn.Module):
    """GatedFusionFormer (v3.0) 調制識別主模型。
    
    該模型融合了 IQ、STFT 及統計特徵，利用 GatingNetwork 動態調整融合比例，
    並通過 Transformer Backbone 進行高級特徵建模。

    Args:
        embed_dim (int): 模型內部的隱藏維度。
        num_classes (int): 分類類別總數。
        stft_time_steps (int): 對齊後的時間序列長度。
        depth (int): Transformer 塊的堆疊深度。
        num_heads (int): 多頭注意力的頭數。
        ffn_act (str): FFN 的激活函數類型。

    Shapes:
        - x_iq, x_std: (B, 2, T_seq)
        - x_stft: (B, 1, F, T_stft)
        - Output: (Logits, Weights) 其中 Logits 為 (B, num_classes), Weights 為 (B, 3)。
    """
    def __init__(self, embed_dim: int = 96, num_classes: int = 11, stft_time_steps: int = 128, 
                 depth: int = 4, num_heads: int = 4, ffn_act: str = "geglu"):
        super().__init__()
        self.embed_iq = nn.Conv1d(2, embed_dim, kernel_size=7, padding=3, bias=False)
        self.embed_stft = StftEncoder2D(in_ch=1, embed_dim=embed_dim)
        self.embed_std = nn.Conv1d(2, embed_dim, kernel_size=7, padding=3, bias=False)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(stft_time_steps)
        
        self.gating_network = GatingNetwork(input_dim=embed_dim)
        # 這裡建議後續將 ffn_act 傳入 FusionTransformerBlock
        self.backbone = nn.Sequential(*[
            FusionTransformerBlock(dim=embed_dim, num_heads=num_heads) for _ in range(depth)
        ])
        self.norm = LayerNorm(embed_dim, eps=1e-6, data_format="channels_first")
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x_iq: torch.Tensor, x_stft: torch.Tensor, x_std: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """多模態特徵融合與分類前向傳播。"""
        feat_iq_raw = self.embed_iq(x_iq)
        feat_stft   = self.embed_stft(x_stft)
        feat_std_raw = self.embed_std(x_std)
    
        feat_iq  = self.adaptive_pool(feat_iq_raw)
        feat_std = self.adaptive_pool(feat_std_raw)
    
        # 獲取門控權重
        weights = self.gating_network(feat_iq, feat_stft, feat_std)
        w_iq, w_stft, w_std = weights[:, 0:1], weights[:, 1:2], weights[:, 2:3]
    
        # 動態加權融合 (Weighted Sum)
        fused_feature = (
            feat_iq   * w_iq.unsqueeze(-1) +
            feat_stft * w_stft.unsqueeze(-1) +
            feat_std  * w_std.unsqueeze(-1)
        )
    
        x = self.backbone(fused_feature)
        x = self.norm(x)
        x = self.avgpool(x).squeeze(-1)
        logits = self.head(x)
        return logits, weights
