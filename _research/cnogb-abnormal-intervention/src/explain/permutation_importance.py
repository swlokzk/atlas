from __future__ import annotations

from typing import Callable

import numpy as np


def _permute_feature(X: np.ndarray, feature_index: int, rng: np.random.Generator) -> np.ndarray:
    X_permuted = np.array(X, copy=True)
    if X_permuted.ndim == 2:
        order = rng.permutation(X_permuted.shape[0])
        X_permuted[:, feature_index] = X_permuted[order, feature_index]
        return X_permuted
    if X_permuted.ndim == 3:
        order = rng.permutation(X_permuted.shape[0])
        X_permuted[:, :, feature_index] = X_permuted[order, :, feature_index]
        return X_permuted
    raise ValueError("Permutation importance expects a 2D or 3D feature array")


def calculate_permutation_importance(
    predict_fn: Callable,
    X: np.ndarray,
    y: np.ndarray,
    metric_fn: Callable,
    n_repeats: int = 5,
    random_state: int = 42,
) -> np.ndarray:
    baseline = metric_fn(y, predict_fn(X))
    rng = np.random.default_rng(random_state)
    importances = []
    for feature_index in range(X.shape[-1]):
        scores = []
        for _ in range(n_repeats):
            X_permuted = _permute_feature(X, feature_index, rng)
            scores.append(metric_fn(y, predict_fn(X_permuted)))
        importances.append(float(baseline - np.mean(scores)))
    return np.asarray(importances, dtype=float)
