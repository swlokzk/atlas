"""Initial scaffold for TS2Vec-style experiments."""

from dataclasses import dataclass
from typing import Optional

from src.models._shared import BackboneConfig, build_backbone_model


@dataclass
class TS2VecConfig(BackboneConfig):
    projection_dim: int = 128


def build_ts2vec(feature_dim: int, config: Optional[TS2VecConfig] = None, device: Optional[str] = None):
    return build_backbone_model(feature_dim, config or TS2VecConfig(num_layers=3), device)


__all__ = ["TS2VecConfig", "build_ts2vec"]
