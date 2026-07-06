from __future__ import annotations

from typing import Callable, Tuple

import numpy as np
import torch


def _extract_prediction(model: torch.nn.Module, batch: torch.Tensor, output, prediction_fn: Callable | None):
    if prediction_fn is not None:
        return prediction_fn(model, batch, output)
    if isinstance(output, dict):
        if hasattr(model, "compute_anomaly_score"):
            return model.compute_anomaly_score(output, batch)
        if "normal_probability" in output:
            return 1.0 - output["normal_probability"].squeeze(-1)
        if "reconstruction" in output:
            return output["reconstruction"]
        raise TypeError("Unsupported model output for MC dropout prediction")
    return output


def predict_with_mc_dropout(
    model: torch.nn.Module,
    loader,
    device: str = "cpu",
    n_samples: int = 20,
    prediction_fn: Callable | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    was_training = model.training
    model.train()
    predictions = []
    with torch.no_grad():
        for _ in range(n_samples):
            sample_predictions = []
            for batch in loader:
                x = batch[0] if isinstance(batch, (tuple, list)) else batch
                x = x.to(device)
                output = model(x)
                pred = _extract_prediction(model, x, output, prediction_fn)
                if isinstance(pred, torch.Tensor):
                    pred = pred.detach().cpu().numpy()
                sample_predictions.append(np.asarray(pred))
            predictions.append(np.concatenate(sample_predictions, axis=0))
    if not was_training:
        model.eval()
    stacked = np.stack(predictions, axis=0)
    return np.mean(stacked, axis=0), np.std(stacked, axis=0)
