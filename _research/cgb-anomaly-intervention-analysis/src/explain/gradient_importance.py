"""Gradient-based feature importance for sequence models."""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import torch


def compute_gradient_importance(
    model: torch.nn.Module,
    x: torch.Tensor,
    target: Optional[torch.Tensor] = None,
    score_fn: Optional[Callable] = None,
    device: str = "cpu",
) -> np.ndarray:
    model.eval()
    batch = x.detach().clone().to(device)
    batch.requires_grad_(True)
    output = model(batch)

    if score_fn is not None:
        objective = score_fn(model, batch, output, target)
    elif isinstance(output, dict) and target is not None and "reconstruction" in output:
        target_tensor = target.to(device)
        objective = torch.mean(torch.abs(output["reconstruction"] - target_tensor))
    elif isinstance(output, dict) and hasattr(model, "compute_anomaly_score"):
        objective = model.compute_anomaly_score(output, batch).mean()
    elif isinstance(output, torch.Tensor):
        objective = output.mean()
    else:
        raise TypeError("Unsupported model output for gradient importance")

    objective.backward()
    return batch.grad.detach().abs().mean(dim=tuple(range(batch.dim() - 1))).cpu().numpy()


__all__ = ["compute_gradient_importance"]