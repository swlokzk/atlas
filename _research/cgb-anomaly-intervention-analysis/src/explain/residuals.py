"""Residual summaries for reconstruction-based anomaly detection."""

from __future__ import annotations

import numpy as np


def analyze_residuals(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    residuals = np.asarray(y_true) - np.asarray(y_pred)
    absolute_residuals = np.abs(residuals)
    summary = {
        "mean": float(np.mean(residuals)),
        "std": float(np.std(residuals)),
        "min": float(np.min(residuals)),
        "max": float(np.max(residuals)),
        "median": float(np.median(residuals)),
        "mae": float(np.mean(absolute_residuals)),
    }
    if residuals.ndim >= 2:
        reduction_axes = tuple(range(residuals.ndim - 1))
        summary["per_feature_mae"] = np.mean(absolute_residuals, axis=reduction_axes).tolist()
    if residuals.ndim == 3:
        summary["per_timestep_mae"] = np.mean(absolute_residuals, axis=(0, 2)).tolist()
    return summary


__all__ = ["analyze_residuals"]