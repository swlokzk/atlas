"""Production-only wrapper for the canonical checkpoint-backed GFF model."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from deploy.checkpoint import CheckpointInfo, download_checkpoint, load_checkpoint_strict
from src.models.model import GatedFusionFormer


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_CONFIG_PATH = PROJECT_ROOT / "deploy" / "config" / "model_config.json"


class GFFInferenceWrapper(torch.nn.Module):
    """Stable all-modality inference interface returning logits and `[iq, stft, std]` weights."""

    def __init__(self, model: GatedFusionFormer):
        super().__init__()
        self.model = model
        self.model.eval()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Path | None = None,
        device: str | torch.device = "cpu",
        config_path: Path = MODEL_CONFIG_PATH,
    ) -> tuple["GFFInferenceWrapper", dict[str, object], CheckpointInfo]:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        model = GatedFusionFormer(
            embed_dim=int(config["embed_dim"]),
            num_classes=int(config["num_classes"]),
            stft_time_steps=int(config["stft_time_steps"]),
            depth=int(config["depth"]),
            num_heads=int(config["num_heads"]),
        )
        path = checkpoint_path or download_checkpoint(
            filename=str(config["checkpoint_filename"])
        )
        info = load_checkpoint_strict(model, path, map_location=device)
        model.to(device)
        return cls(model).to(device), config, info

    def forward(
        self,
        iq: torch.Tensor,
        stft: torch.Tensor,
        std: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model(iq, stft, std)

    @staticmethod
    def validate_outputs(logits: torch.Tensor, gating_weights: torch.Tensor) -> None:
        """Validate production telemetry outside the ONNX-exported forward graph."""
        if not torch.isfinite(logits).all() or not torch.isfinite(gating_weights).all():
            raise RuntimeError("Model produced non-finite inference outputs.")
        if torch.any(gating_weights < 0):
            raise RuntimeError("Model produced negative gating weights.")
        if not torch.allclose(
            gating_weights.sum(dim=1),
            torch.ones(gating_weights.shape[0], device=gating_weights.device),
            atol=1e-5,
        ):
            raise RuntimeError("Model gating weights do not sum to one.")