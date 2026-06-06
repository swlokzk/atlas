"""TranAD-style sequence reconstruction model for anomaly detection."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from src.models._shared import BackboneConfig
from src.models.tranad.discriminator import Discriminator
from src.models.tranad.loss import TranADLoss


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


@dataclass
class TranADConfig(BackboneConfig):
    adversarial_weight: float = 1.0
    reconstruction_weight: float = 1.0
    discriminator_hidden: int = 64


class TranAD(nn.Module):
    """Sequence autoencoder with an auxiliary discriminator."""

    def __init__(self, feature_dim: int, config: Optional[TranADConfig] = None, device: Optional[str] = None):
        super().__init__()
        self.feature_dim = feature_dim
        self.config = config or TranADConfig(num_layers=3)
        self.device_name = device or self.config.device

        self.input_projection = nn.Linear(feature_dim, self.config.d_model)
        self.positional_encoding = PositionalEncoding(self.config.d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.config.d_model,
            nhead=self.config.nhead,
            dim_feedforward=self.config.dim_feedforward,
            dropout=self.config.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.config.num_layers)
        self.reconstruction_head = nn.Sequential(
            nn.Linear(self.config.d_model, self.config.d_model),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.d_model, feature_dim),
        )
        self.discriminator = Discriminator(
            d_model=self.config.d_model,
            hidden_dim=self.config.discriminator_hidden,
            dropout=self.config.dropout,
        )
        self.loss_fn = TranADLoss(
            adversarial_weight=self.config.adversarial_weight,
            reconstruction_weight=self.config.reconstruction_weight,
        )

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.input_projection(x)
        embedded = self.positional_encoding(embedded)
        return self.encoder(embedded)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.reconstruction_head(latent)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = x.to(self.device)
        latent = self.encode(x)
        reconstruction = self.decode(latent)
        normal_probability = self.discriminator(latent)
        return {
            "x_recon": reconstruction,
            "reconstruction": reconstruction,
            "z": latent,
            "latent": latent,
            "disc_pred": normal_probability,
            "normal_probability": normal_probability,
        }

    def compute_losses(
        self,
        output: Dict[str, torch.Tensor],
        x_true: torch.Tensor,
        negative_probability: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        normal_probability = output["normal_probability"]
        reconstruction = output["reconstruction"]
        generator_loss, generator_metrics = self.loss_fn.generator_loss(x_true, reconstruction, normal_probability)
        if negative_probability is None:
            negative_probability = torch.zeros_like(normal_probability)
        discriminator_loss, discriminator_metrics = self.loss_fn.discriminator_loss(normal_probability, negative_probability)
        return generator_loss, discriminator_loss, {**generator_metrics, **discriminator_metrics}

    def compute_anomaly_score(self, output: Dict[str, torch.Tensor], x_true: torch.Tensor) -> torch.Tensor:
        return self.loss_fn.anomaly_score(x_true, output["reconstruction"], output["normal_probability"])

    def detect_anomalies(self, x: torch.Tensor, threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        self.eval()
        with torch.no_grad():
            output = self.forward(x)
            scores = self.compute_anomaly_score(output, x.to(self.device))
            min_score = scores.min()
            max_score = scores.max()
            normalized = (scores - min_score) / (max_score - min_score + 1e-8)
            labels = (normalized >= threshold).long()
        return normalized.cpu().numpy(), labels.cpu().numpy()


def build_tranad(feature_dim: int, config: Optional[TranADConfig] = None, device: Optional[str] = None) -> TranAD:
    model = TranAD(feature_dim, config=config, device=device)
    return model.to(device or (config.device if config is not None else "cpu"))


__all__ = ["TranAD", "TranADConfig", "build_tranad"]
