"""Initial scaffold for TimesNet-style experiments."""

from dataclasses import dataclass
from typing import Optional

from src.models._shared import BackboneConfig, build_backbone_model


@dataclass
class TimesNetConfig(BackboneConfig):
    num_kernels: int = 6


def build_timesnet(feature_dim: int, config: Optional[TimesNetConfig] = None, device: Optional[str] = None):
    return build_backbone_model(feature_dim, config or TimesNetConfig(num_layers=3), device)


__all__ = ["TimesNetConfig", "build_timesnet"]
