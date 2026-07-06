"""Shared scaffolding used by model-family placeholders and initial baselines."""

from dataclasses import dataclass
from typing import Optional

from src.models.transformer import build_model as build_transformer_backbone


@dataclass
class BackboneConfig:
    d_model: int = 128
    nhead: int = 4
    num_layers: int = 2
    dim_feedforward: int = 256
    dropout: float = 0.1
    device: str = "cpu"


def build_backbone_model(feature_dim: int, config: BackboneConfig, device: Optional[str] = None):
    params = {
        "D_MODEL": config.d_model,
        "N_HEAD": config.nhead,
        "NUM_LAYERS": config.num_layers,
    }
    return build_transformer_backbone(params, feature_dim, device=device or config.device)


__all__ = ["BackboneConfig", "build_backbone_model"]
