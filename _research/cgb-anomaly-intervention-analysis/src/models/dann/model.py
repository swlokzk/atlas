"""Initial scaffold for DANN-style transfer experiments."""

from dataclasses import dataclass
from typing import Optional

from src.models._shared import BackboneConfig, build_backbone_model


@dataclass
class DANNConfig(BackboneConfig):
    domain_loss_weight: float = 1.0


def build_dann(feature_dim: int, config: Optional[DANNConfig] = None, device: Optional[str] = None):
    return build_backbone_model(feature_dim, config or DANNConfig(num_layers=3), device)


__all__ = ["DANNConfig", "build_dann"]
