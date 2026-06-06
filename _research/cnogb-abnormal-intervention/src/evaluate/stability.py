"""Stability summaries across repeated experiment runs."""

from __future__ import annotations

from typing import Dict, Iterable

import numpy as np


def _collect_metric(results_list: Iterable[Dict[str, float]], key: str) -> np.ndarray:
    values = [float(item[key]) for item in results_list if key in item and item[key] is not None]
    return np.asarray(values, dtype=float)


def compute_stability_metrics(results_list: list[dict], metric_keys: Iterable[str] = ("auroc", "auprc", "f1")) -> dict:
    if not results_list:
        return {"n_runs": 0}
    summary = {"n_runs": len(results_list)}
    for key in metric_keys:
        values = _collect_metric(results_list, key)
        if values.size == 0:
            continue
        summary[f"{key}_mean"] = float(values.mean())
        summary[f"{key}_std"] = float(values.std(ddof=0))
        summary[f"{key}_min"] = float(values.min())
        summary[f"{key}_max"] = float(values.max())
        summary[f"{key}_range"] = float(values.max() - values.min())
    return summary


__all__ = ["compute_stability_metrics"]