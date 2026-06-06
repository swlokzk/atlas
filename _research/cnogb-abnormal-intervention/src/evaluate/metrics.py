"""Metrics for anomaly detection experiments."""

from __future__ import annotations

from typing import Dict, Iterable

import numpy as np
from sklearn.metrics import auc, confusion_matrix, f1_score, precision_recall_curve, precision_recall_fscore_support, roc_auc_score, roc_curve


def compute_auroc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y_true, y_scores))
    except ValueError:
        return 0.5


def compute_auprc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    try:
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        return float(auc(recall, precision))
    except ValueError:
        return 0.0


def compute_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    try:
        return float(f1_score(y_true, y_pred, zero_division=0))
    except ValueError:
        return 0.0


def compute_precision_at_k(y_true: np.ndarray, y_scores: np.ndarray, k_values: Iterable[int] | None = None) -> Dict[int, float]:
    if k_values is None:
        k_values = (10, 20, 50)
    sorted_labels = y_true[np.argsort(-y_scores)]
    results: Dict[int, float] = {}
    for k in k_values:
        if k <= 0 or k > len(sorted_labels):
            continue
        results[int(k)] = float(np.mean(sorted_labels[:k]))
    return results


def compute_recall_at_fpr(y_true: np.ndarray, y_scores: np.ndarray, fpr_threshold: float = 0.1) -> float:
    try:
        fpr, tpr, _ = roc_curve(y_true, y_scores)
    except ValueError:
        return 0.0
    index = int(np.argmin(np.abs(fpr - fpr_threshold)))
    return float(tpr[index])


def compute_threshold_metrics(y_true: np.ndarray, y_scores: np.ndarray, threshold: float) -> Dict[str, float]:
    y_pred = (y_scores >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "specificity": float(specificity),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def compute_anomaly_metrics(y_true: np.ndarray, y_scores: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    metrics = {
        "auroc": compute_auroc(y_true, y_scores),
        "auprc": compute_auprc(y_true, y_scores),
        "recall_at_10_fpr": compute_recall_at_fpr(y_true, y_scores, 0.1),
    }
    metrics.update({f"precision_at_{k}": value for k, value in compute_precision_at_k(y_true, y_scores).items()})
    metrics.update(compute_threshold_metrics(y_true, y_scores, threshold))
    return metrics


__all__ = [
    "compute_anomaly_metrics",
    "compute_auprc",
    "compute_auroc",
    "compute_f1",
    "compute_precision_at_k",
    "compute_recall_at_fpr",
    "compute_threshold_metrics",
]