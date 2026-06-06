"""Runtime and model-size helpers for experiment reporting."""

from __future__ import annotations

import time
from typing import Optional

import torch


def _synchronize_if_needed(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def measure_efficiency(
    model: torch.nn.Module,
    sample_batch: Optional[torch.Tensor] = None,
    wall_time_seconds: Optional[float] = None,
    repeats: int = 10,
) -> dict:
    device = next(model.parameters()).device
    metrics = {
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "trainable_parameter_count": int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)),
        "device": str(device),
    }
    if wall_time_seconds is not None:
        metrics["wall_time_seconds"] = float(wall_time_seconds)

    if sample_batch is None:
        return metrics

    batch = sample_batch.to(device)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for _ in range(min(2, repeats)):
            _ = model(batch)
        _synchronize_if_needed(device)
        start = time.perf_counter()
        for _ in range(repeats):
            _ = model(batch)
        _synchronize_if_needed(device)
        elapsed = time.perf_counter() - start
    if was_training:
        model.train()

    metrics["mean_inference_ms"] = float(elapsed * 1000.0 / max(1, repeats))
    metrics["batch_size"] = int(batch.shape[0])
    return metrics


__all__ = ["measure_efficiency"]