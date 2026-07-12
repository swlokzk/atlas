"""Strict checkpoint loading helpers for deployment gates."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

import torch


_STATE_DICT_KEYS = ("state_dict", "model_state_dict", "model")


def sha256_file(path: str | Path) -> str:
    """Return the SHA256 digest of a checkpoint file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as checkpoint:
        for chunk in iter(lambda: checkpoint.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_state_dict(checkpoint: Any) -> Mapping[str, torch.Tensor]:
    """Extract a state dict from a raw or conventionally wrapped checkpoint."""
    if not isinstance(checkpoint, Mapping):
        raise ValueError(
            "Unsupported checkpoint format: expected a state_dict mapping or a "
            f"mapping containing one of {_STATE_DICT_KEYS}."
        )

    if checkpoint and all(isinstance(value, torch.Tensor) for value in checkpoint.values()):
        return checkpoint

    for key in _STATE_DICT_KEYS:
        candidate = checkpoint.get(key)
        if isinstance(candidate, Mapping) and candidate and all(
            isinstance(value, torch.Tensor) for value in candidate.values()
        ):
            return candidate

    raise ValueError(
        "Unsupported checkpoint dictionary: no tensor state_dict was found under "
        f"{_STATE_DICT_KEYS}. Inspect the checkpoint before changing the model."
    )


def load_checkpoint_strict(
    model: torch.nn.Module, checkpoint_path: str | Path
) -> tuple[torch.nn.Module, str]:
    """Load a checkpoint strictly and return the model plus its SHA256 digest."""
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}. Export is blocked until a compatible "
            "checkpoint is supplied."
        )

    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        state_dict = extract_state_dict(checkpoint)
        model.load_state_dict(state_dict, strict=True)
    except (RuntimeError, ValueError) as error:
        raise RuntimeError(
            f"Checkpoint {path} is incompatible with {type(model).__module__}."
            f"{type(model).__qualname__}: {error}"
        ) from error

    model.eval()
    return model, sha256_file(path)
