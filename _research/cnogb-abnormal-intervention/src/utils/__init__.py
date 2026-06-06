"""Utility helpers shared by scripts and experiments."""

from __future__ import annotations

import random

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - torch may be missing in lightweight validation environments
    torch = None

from .visualization import plot_predictions, set_matplotlib_defaults


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is None:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


__all__ = ["plot_predictions", "set_matplotlib_defaults", "set_seed"]