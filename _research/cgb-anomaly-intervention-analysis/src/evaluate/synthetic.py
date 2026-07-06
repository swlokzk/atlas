"""Synthetic anomaly injection helpers for validation."""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np


def _safe_scale(data: np.ndarray) -> np.ndarray:
    scale = np.std(data, axis=0)
    return np.where(scale == 0, 1.0, scale)


def inject_spike(data: np.ndarray, indices: Iterable[int], amplitude: float = 3.0, random_state: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    result = np.array(data, copy=True)
    scale = _safe_scale(result)
    for index in indices:
        if 0 <= index < len(result):
            sign = rng.choice([-1.0, 1.0])
            result[index] = result[index] + sign * amplitude * scale
    return result


def inject_level_shift(data: np.ndarray, start_idx: int, end_idx: int, shift: float = 2.0) -> np.ndarray:
    result = np.array(data, copy=True)
    scale = _safe_scale(result)
    result[max(0, start_idx) : min(len(result), end_idx)] = result[max(0, start_idx) : min(len(result), end_idx)] + shift * scale
    return result


def inject_trend_change(data: np.ndarray, start_idx: int, end_idx: int, trend_slope: float = 0.1) -> np.ndarray:
    result = np.array(data, copy=True)
    upper = min(len(result), end_idx)
    for offset, index in enumerate(range(max(0, start_idx), upper)):
        result[index] = result[index] + trend_slope * offset
    return result


def create_synthetic_test_set(
    data: np.ndarray,
    anomaly_ratio: float = 0.1,
    types: Iterable[str] | None = None,
    random_seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_seed)
    anomaly_types = tuple(types or ("spike", "level_shift", "trend_change"))
    result = np.array(data, copy=True)
    labels = np.zeros(len(result), dtype=int)
    n_samples = len(result)
    n_spikes = max(1, int(n_samples * anomaly_ratio / max(1, len(anomaly_types))))
    segment_length = max(2, min(20, n_samples // 10 if n_samples > 10 else n_samples))

    if "spike" in anomaly_types:
        spike_indices = rng.choice(n_samples, size=min(n_spikes, n_samples), replace=False)
        result = inject_spike(result, spike_indices, random_state=random_seed)
        labels[spike_indices] = 1

    if "level_shift" in anomaly_types and n_samples >= 2:
        start_idx = int(rng.integers(0, max(1, n_samples - segment_length + 1)))
        end_idx = min(n_samples, start_idx + segment_length)
        result = inject_level_shift(result, start_idx, end_idx)
        labels[start_idx:end_idx] = 1

    if "trend_change" in anomaly_types and n_samples >= 2:
        start_idx = int(rng.integers(0, max(1, n_samples - segment_length + 1)))
        end_idx = min(n_samples, start_idx + segment_length)
        result = inject_trend_change(result, start_idx, end_idx)
        labels[start_idx:end_idx] = 1

    return result, labels


__all__ = [
    "create_synthetic_test_set",
    "inject_level_shift",
    "inject_spike",
    "inject_trend_change",
]