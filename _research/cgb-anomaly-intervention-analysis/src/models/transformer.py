import math
import torch
import torch.nn as nn
from typing import Optional


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :]
        return x


class TransformerPredictor(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.linear_in = nn.Linear(feature_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.linear_out = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, features)
        x = self.linear_in(x)
        x = self.pos_encoder(x)
        x = x.permute(1, 0, 2)  # seq_len, batch, d_model
        h = self.transformer_encoder(x)
        h = h[-1]  # last time step
        out = self.linear_out(h)
        return out.squeeze(-1)


def build_model(params: dict, feature_dim: int, device: Optional[str] = "cpu"):
    model = TransformerPredictor(feature_dim, d_model=params.get("D_MODEL", 128), nhead=params.get("N_HEAD", 4), num_layers=params.get("NUM_LAYERS", 2))
    return model.to(device)
