"""Initial scaffold for DADA-style experiments."""

from dataclasses import dataclass
from typing import Optional

from src.models._shared import BackboneConfig, build_backbone_model


@dataclass
class DADAConfig(BackboneConfig):
    bottleneck_dim: int = 64


def build_dada(feature_dim: int, config: Optional[DADAConfig] = None, device: Optional[str] = None):
    return build_backbone_model(feature_dim, config or DADAConfig(num_layers=3), device)


__all__ = ["DADAConfig", "build_dada"]
