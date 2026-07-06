"""Initial scaffold for PatchTST-style experiments."""

from dataclasses import dataclass
from typing import Optional

from src.models._shared import BackboneConfig, build_backbone_model


@dataclass
class PatchTSTConfig(BackboneConfig):
    patch_len: int = 8
    stride: int = 4


def build_patchtst(feature_dim: int, config: Optional[PatchTSTConfig] = None, device: Optional[str] = None):
    return build_backbone_model(feature_dim, config or PatchTSTConfig(num_layers=3), device)


__all__ = ["PatchTSTConfig", "build_patchtst"]
