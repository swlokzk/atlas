"""Calibration diagnostics for anomaly scores treated as probabilities."""

from __future__ import annotations

import numpy as np


def compute_calibration_metrics(y_true: np.ndarray, y_scores: np.ndarray, n_bins: int = 10) -> dict:
    y_true = np.asarray(y_true).astype(float)
    y_scores = np.clip(np.asarray(y_scores).astype(float), 0.0, 1.0)
    if y_true.size == 0:
        return {"ece": 0.0, "mce": 0.0, "bin_stats": []}

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_scores, bin_edges[1:-1], right=True)
    ece = 0.0
    mce = 0.0
    bin_stats = []

    for bin_index in range(n_bins):
        mask = bin_ids == bin_index
        if not np.any(mask):
            continue
        observed = float(np.mean(y_true[mask]))
        confidence = float(np.mean(y_scores[mask]))
        gap = abs(observed - confidence)
        weight = float(np.mean(mask))
        ece += gap * weight
        mce = max(mce, gap)
        bin_stats.append(
            {
                "bin_index": int(bin_index),
                "count": int(mask.sum()),
                "accuracy": observed,
                "confidence": confidence,
                "gap": float(gap),
            }
        )

    return {"ece": float(ece), "mce": float(mce), "bin_stats": bin_stats}


__all__ = ["compute_calibration_metrics"]