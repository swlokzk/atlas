"""Initial scaffold for the Anomaly Transformer family."""

from dataclasses import dataclass
from typing import Optional

from src.models._shared import BackboneConfig, build_backbone_model


@dataclass
class AnomalyTransformerConfig(BackboneConfig):
    window_size: int = 16


def build_abnormal_transformer(
    feature_dim: int,
    config: Optional[AnomalyTransformerConfig] = None,
    device: Optional[str] = None,
):
    return build_backbone_model(feature_dim, config or AnomalyTransformerConfig(), device)


__all__ = ["AnomalyTransformerConfig", "build_abnormal_transformer"]
