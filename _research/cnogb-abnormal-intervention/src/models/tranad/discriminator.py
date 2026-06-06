"""Adversarial discriminators for sequence anomaly detection."""

from __future__ import annotations

import torch
import torch.nn as nn


class Discriminator(nn.Module):
    """Classify whether a latent representation looks normal."""

    def __init__(self, d_model: int = 128, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.dim() == 3:
            latent = latent.mean(dim=1)
        return self.network(latent)


class TransformerDiscriminator(nn.Module):
    """Sequence-aware discriminator variant kept for experimentation."""

    def __init__(self, d_model: int = 128, nhead: int = 4, num_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=max(4 * d_model, 128),
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, 1), nn.Sigmoid())

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(latent)
        pooled = encoded.mean(dim=1)
        return self.head(pooled)


__all__ = ["Discriminator", "TransformerDiscriminator"]