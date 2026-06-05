"""L1 Cross-domain validation: Leave-one-out across CDB, ADBC, EXIM bonds.

This module implements the "跨標的一致性驗證" (L1) evaluation framework.
For each LOO round, two bonds are used for training and one for testing,
verifying that anomaly signals are *systematic* rather than bond-specific.
"""

from __future__ import annotations

import time
import tracemalloc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import MinMaxScaler

from src.data.dataset import build_dataloaders
from src.data.features import prepare_pair_feature_artifacts
from src.data.loader import find_paired_files
from src.data.processing import process_pair
from src.data.sequence import create_sequences
from src.explain.permutation_importance import calculate_permutation_importance
from src.explain.uncertainty import predict_with_mc_dropout
from src.models.tranad import TranADConfig, build_tranad


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _mse_per_sample(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Return per-sample MSE (scalar error per time step)."""
    diff = y_true.ravel() - y_pred.ravel()
    return diff ** 2


def _normalize_scores(scores: np.ndarray) -> np.ndarray:
    """Min-max normalize anomaly scores to [0, 1]."""
    s_min, s_max = scores.min(), scores.max()
    if s_max - s_min < 1e-10:
        return np.zeros_like(scores)
    return (scores - s_min) / (s_max - s_min)


def _compute_f1_at_threshold(
    anomaly_scores: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5,
) -> float:
    """Compute F1 score at a fixed anomaly threshold."""
    preds = (anomaly_scores >= threshold).astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _compute_ece(scores: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE)."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(scores)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (scores >= lo) & (scores < hi)
        if mask.sum() == 0:
            continue
        avg_conf = scores[mask].mean()
        avg_acc = labels[mask].mean()
        ece += mask.sum() / n * abs(avg_conf - avg_acc)
    return float(ece)


# ---------------------------------------------------------------------------
# Standalone function (used by unit tests)
# ---------------------------------------------------------------------------

def compute_anomaly_consistency(results: Dict[str, dict]) -> Dict:
    """Compute cross-bond anomaly consistency metrics.

    Parameters
    ----------
    results:
        Dict keyed by bond name (``'cdb'``, ``'adbc'``, ``'exim'``).
        Each value must contain ``'anomaly_scores'`` (np.ndarray).

    Returns
    -------
    dict with keys:
        - ``correlation``:  pairwise Pearson correlations
        - ``consistency_ratio``:  fraction of time steps where **all three**
          bonds simultaneously exceed the 0.5 threshold
        - ``mean_pairwise_corr``:  average of the three pairwise correlations
    """
    bond_names = list(results.keys())
    scores = {b: np.asarray(results[b]["anomaly_scores"]).ravel() for b in bond_names}

    # Align lengths to the shortest series
    min_len = min(len(s) for s in scores.values())
    scores = {b: s[:min_len] for b, s in scores.items()}

    # Pairwise Pearson correlations
    correlation: Dict[str, float] = {}
    for i, b1 in enumerate(bond_names):
        for b2 in bond_names[i + 1 :]:
            key = f"{b1}_{b2}"
            r, _ = stats.pearsonr(scores[b1], scores[b2])
            correlation[key] = float(r)

    # Simultaneous anomaly ratio (threshold = 0.5)
    threshold = 0.5
    is_anomaly = {b: scores[b] >= threshold for b in bond_names}
    all_anomaly = np.ones(min_len, dtype=bool)
    for b in bond_names:
        all_anomaly &= is_anomaly[b]
    consistency_ratio = float(all_anomaly.mean())

    mean_pairwise_corr = float(np.mean(list(correlation.values()))) if correlation else 0.0

    return {
        "correlation": correlation,
        "consistency_ratio": consistency_ratio,
        "mean_pairwise_corr": mean_pairwise_corr,
    }


# ---------------------------------------------------------------------------
# Main validator class
# ---------------------------------------------------------------------------

@dataclass
class LOORoundResult:
    """Result for a single Leave-one-out round."""
    test_bond: str
    train_bonds: List[str]
    anomaly_scores: np.ndarray
    uncertainty: np.ndarray
    auroc: float
    auprc: float
    f1: float
    feature_importance: Dict[str, float]
    residuals: np.ndarray
    n_train_samples: int
    n_test_samples: int


class LeaveOneOutValidator:
    """Leave-one-out cross-domain validator for the three policy-bank bonds.

    Usage::

        validator = LeaveOneOutValidator(config, project_root=Path('.'))
        results = validator.run_loo_experiment()
        consistency = validator.compute_anomaly_consistency(results)
    """

    BONDS: List[str] = ["cdb", "adbc", "exim"]

    def __init__(self, config: dict, project_root: Optional[Path] = None):
        self.config = config
        self.project_root = project_root or Path(__file__).resolve().parents[3]

        data_cfg = config.get("data", {})
        self.raw_dir = self.project_root / data_cfg.get("raw_dir", "data/raw")
        self.time_col = data_cfg.get("time_column", "date")
        self.target_col = data_cfg.get("target_column", "weighted_rate")

        training_cfg = config.get("training", {})
        self.seq_len: int = int(training_cfg.get("sequence_length", 16))
        self.batch_size: int = int(training_cfg.get("batch_size", 64))
        self.num_epochs: int = int(training_cfg.get("num_epochs", 20))
        self.lr: float = float(training_cfg.get("learning_rate", 1e-3))
        self.device: str = str(training_cfg.get("device", "cpu"))

        model_cfg = config.get("model", {})
        self.model_cfg = model_cfg

        eval_cfg = config.get("evaluation", {})
        self.mc_samples: int = int(eval_cfg.get("mc_samples", 20))
        self.stability_seeds: List[int] = list(eval_cfg.get("stability_seeds", [42, 123, 456, 789, 1000]))
        self.anomaly_threshold: float = float(eval_cfg.get("anomaly_threshold", 0.5))

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_data(self, bond_name: str) -> Optional[dict]:
        """Load and process a single bond's data.

        Returns a processed-info dict (from ``process_pair``) with feature
        artifacts attached, or ``None`` if the data files are not found.
        """
        pairs = find_paired_files(self.raw_dir)
        pair = next((p for p in pairs if p.get("name") == bond_name), None)
        if pair is None or not pair.get("train") or not pair.get("exam"):
            return None

        info = process_pair(pair, self.time_col, self.target_col)
        if not info.get("valid"):
            return None

        training_params = {
            "sequence_length": self.seq_len,
            "batch_size": self.batch_size,
        }
        info = prepare_pair_feature_artifacts(info, training_params)
        return info

    def _make_sequences(self, info: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Build (X_train_seq, y_train_seq, X_exam_seq, y_exam_seq)."""
        X_train_seq, y_train_seq = create_sequences(
            info["X_train_transformed_features"],
            info["y_train_scaled"],
            self.seq_len,
        )
        X_exam_seq, y_exam_seq = create_sequences(
            info["X_exam_transformed_features"],
            info["y_exam_scaled"],
            self.seq_len,
        )
        return X_train_seq, y_train_seq, X_exam_seq, y_exam_seq

    # ------------------------------------------------------------------
    # Model building & training
    # ------------------------------------------------------------------

    def _build_model(self, feature_dim: int) -> nn.Module:
        cfg = self.model_cfg
        tranad_cfg = TranADConfig(
            d_model=int(cfg.get("d_model", 128)),
            nhead=int(cfg.get("nhead", 4)),
            num_layers=int(cfg.get("num_layers", 3)),
            dim_feedforward=int(cfg.get("dim_feedforward", 256)),
            dropout=float(cfg.get("dropout", 0.1)),
            device=self.device,
            adversarial_weight=float(cfg.get("adversarial_weight", 1.0)),
        )
        return build_tranad(feature_dim, config=tranad_cfg, device=self.device)

    def _train(
        self,
        model: nn.Module,
        X_seq: np.ndarray,
        y_seq: np.ndarray,
        seed: int = 42,
    ) -> nn.Module:
        """Train ``model`` on the provided sequences."""
        torch.manual_seed(seed)
        train_loader, _ = build_dataloaders(X_seq, y_seq, self.batch_size)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=self.lr)

        model.train()
        for _ in range(self.num_epochs):
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                optimizer.zero_grad()
                pred = model(x_batch)
                loss = criterion(pred, y_batch.squeeze(-1))
                loss.backward()
                optimizer.step()
        return model

    # ------------------------------------------------------------------
    # Anomaly scoring
    # ------------------------------------------------------------------

    def _get_predictions(self, model: nn.Module, X_seq: np.ndarray) -> np.ndarray:
        """Run model in eval mode and return predictions (numpy)."""
        model.eval()
        preds: List[np.ndarray] = []
        batch_size = self.batch_size
        with torch.no_grad():
            for start in range(0, len(X_seq), batch_size):
                batch = torch.from_numpy(X_seq[start : start + batch_size]).float().to(self.device)
                out = model(batch).cpu().numpy()
                preds.append(out)
        return np.concatenate(preds, axis=0)

    def _anomaly_scores_from_errors(
        self,
        model: nn.Module,
        X_seq: np.ndarray,
        y_seq: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (normalized_anomaly_scores, raw_residuals)."""
        preds = self._get_predictions(model, X_seq)
        residuals = _mse_per_sample(y_seq.ravel(), preds)
        scores = _normalize_scores(residuals)
        return scores, residuals

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------

    def _compute_feature_importance(
        self,
        model: nn.Module,
        X_exam_seq: np.ndarray,
        y_exam_seq: np.ndarray,
        feature_columns: List[str],
    ) -> Dict[str, float]:
        """Compute permutation importance over exam sequences.

        ``X_exam_seq`` has shape ``(n_samples, seq_len, n_features)``.
        We flatten the feature dimension for permutation.
        """
        n_features = X_exam_seq.shape[-1]

        def predict_fn(X_flat: np.ndarray) -> np.ndarray:
            # X_flat: (n_samples, seq_len * n_features) – unflatten & predict
            X_3d = X_flat.reshape(-1, self.seq_len, n_features)
            return self._get_predictions(model, X_3d)

        X_flat = X_exam_seq.reshape(len(X_exam_seq), -1)
        y_flat = y_exam_seq.ravel()

        def metric_fn(y_true: np.ndarray, y_pred: np.ndarray) -> float:
            return float(np.mean((y_true - y_pred) ** 2))

        # Compute importance per original feature (average over seq_len positions)
        importances_flat = calculate_permutation_importance(predict_fn, X_flat, y_flat, metric_fn, n_repeats=3)
        importances_flat = importances_flat.reshape(self.seq_len, n_features)
        importances_per_feature = importances_flat.mean(axis=0)

        # Build name→importance mapping
        cols = feature_columns if len(feature_columns) == n_features else [f"feature_{i}" for i in range(n_features)]
        return {col: float(imp) for col, imp in zip(cols, importances_per_feature)}

    # ------------------------------------------------------------------
    # Main LOO experiment
    # ------------------------------------------------------------------

    def run_loo_experiment(self, seed: int = 42) -> Dict[str, dict]:
        """Run 3-fold Leave-one-out experiment.

        Returns
        -------
        dict keyed by test bond name (``'cdb'``, ``'adbc'``, ``'exim'``).
        Each value contains:
            anomaly_scores, uncertainty, auroc, auprc, f1,
            feature_importance, residuals, n_train_samples, n_test_samples.
        """
        # Load all bond data up front
        bond_data: Dict[str, Optional[dict]] = {}
        for bond in self.BONDS:
            bond_data[bond] = self.load_data(bond)

        available = [b for b in self.BONDS if bond_data[b] is not None]
        if len(available) < 2:
            raise RuntimeError(
                f"Need at least 2 bonds with data. Found: {available}. "
                f"Ensure data files exist under '{self.raw_dir}'."
            )

        results: Dict[str, dict] = {}

        for test_bond in available:
            train_bonds = [b for b in available if b != test_bond]

            # ---- Merge training sequences from all training bonds ----
            X_train_parts, y_train_parts = [], []
            feature_columns: List[str] = []
            feature_dim: int = 0
            for tb in train_bonds:
                info = bond_data[tb]
                assert info is not None
                X_tr, y_tr, _, _ = self._make_sequences(info)
                X_train_parts.append(X_tr)
                y_train_parts.append(y_tr)
                if feature_dim == 0:
                    feature_dim = X_tr.shape[-1]
                    feature_columns = info.get("feature_columns", [])

            X_train_merged = np.concatenate(X_train_parts, axis=0)
            y_train_merged = np.concatenate(y_train_parts, axis=0)

            # ---- Test sequences from test bond ----
            test_info = bond_data[test_bond]
            assert test_info is not None
            _, _, X_exam_seq, y_exam_seq = self._make_sequences(test_info)
            test_feature_dim = X_exam_seq.shape[-1]

            # Pad or truncate feature dim if there's a mismatch
            if test_feature_dim != feature_dim:
                min_dim = min(feature_dim, test_feature_dim)
                X_train_merged = X_train_merged[:, :, :min_dim]
                X_exam_seq = X_exam_seq[:, :, :min_dim]
                feature_dim = min_dim
                feature_columns = feature_columns[:min_dim]

            # ---- Build + train model ----
            model = self._build_model(feature_dim)
            model = self._train(model, X_train_merged, y_train_merged, seed=seed)

            # ---- Anomaly scores on exam data ----
            anomaly_scores, residuals = self._anomaly_scores_from_errors(
                model, X_exam_seq, y_exam_seq
            )

            # ---- Pseudo-labels: also score training data to build normal baseline ----
            # Label train samples as 0 (normal), exam samples as 1 (potential anomaly)
            train_scores_list: List[np.ndarray] = []
            for tb in train_bonds:
                info = bond_data[tb]
                assert info is not None
                X_tr, y_tr, _, _ = self._make_sequences(info)
                if X_tr.shape[-1] != feature_dim:
                    X_tr = X_tr[:, :, :feature_dim]
                s, _ = self._anomaly_scores_from_errors(model, X_tr, y_tr)
                train_scores_list.append(s)
            train_scores_all = np.concatenate(train_scores_list, axis=0)

            all_scores = np.concatenate([train_scores_all, anomaly_scores])
            all_labels = np.concatenate([
                np.zeros(len(train_scores_all), dtype=int),
                np.ones(len(anomaly_scores), dtype=int),
            ])

            # Subsample for efficiency if needed
            max_eval = 20_000
            if len(all_scores) > max_eval:
                idx = np.random.default_rng(seed).choice(len(all_scores), max_eval, replace=False)
                idx.sort()
                all_scores_eval = all_scores[idx]
                all_labels_eval = all_labels[idx]
            else:
                all_scores_eval = all_scores
                all_labels_eval = all_labels

            if all_labels_eval.sum() == 0 or all_labels_eval.sum() == len(all_labels_eval):
                auroc, auprc, f1 = 0.5, float(all_labels_eval.mean()), 0.0
            else:
                auroc = float(roc_auc_score(all_labels_eval, all_scores_eval))
                auprc = float(average_precision_score(all_labels_eval, all_scores_eval))
                f1 = _compute_f1_at_threshold(
                    all_scores_eval, all_labels_eval, self.anomaly_threshold
                )

            # ---- MC Dropout uncertainty ----
            exam_loader, _ = build_dataloaders(X_exam_seq, y_exam_seq, self.batch_size)
            _, uncertainty = predict_with_mc_dropout(model, exam_loader, self.device, self.mc_samples)
            uncertainty = uncertainty.ravel()

            # ---- Feature importance ----
            feature_importance = self._compute_feature_importance(
                model, X_exam_seq, y_exam_seq, feature_columns
            )

            results[test_bond] = {
                "anomaly_scores": anomaly_scores,
                "uncertainty": uncertainty,
                "auroc": auroc,
                "auprc": auprc,
                "f1": f1,
                "feature_importance": feature_importance,
                "residuals": residuals,
                "n_train_samples": int(len(X_train_merged)),
                "n_test_samples": int(len(X_exam_seq)),
                "train_bonds": train_bonds,
            }

        return results

    # ------------------------------------------------------------------
    # Consistency metric
    # ------------------------------------------------------------------

    def compute_anomaly_consistency(self, results: Dict[str, dict]) -> Dict:
        """Delegate to the module-level ``compute_anomaly_consistency``."""
        return compute_anomaly_consistency(results)

    # ------------------------------------------------------------------
    # Stability test
    # ------------------------------------------------------------------

    def run_stability_test(self, seed_list: Optional[List[int]] = None) -> Dict:
        """Run LOO with multiple seeds to assess stability.

        Returns mean/std of AUROC and a one-sample t-test p-value against 0.5.
        """
        seeds = seed_list or self.stability_seeds
        auroc_per_seed: List[float] = []

        for s in seeds:
            try:
                res = self.run_loo_experiment(seed=s)
                bond_aurocs = [v["auroc"] for v in res.values()]
                auroc_per_seed.append(float(np.mean(bond_aurocs)))
            except Exception:
                pass

        if not auroc_per_seed:
            return {"mean": 0.5, "std": 0.0, "p_value": 1.0, "n_runs": 0}

        arr = np.array(auroc_per_seed)
        t_stat, p_value = stats.ttest_1samp(arr, popmean=0.5)
        return {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "p_value": float(p_value),
            "n_runs": len(arr),
            "auroc_per_seed": auroc_per_seed,
        }

    # ------------------------------------------------------------------
    # Efficiency measurement
    # ------------------------------------------------------------------

    def measure_efficiency(self, seed: int = 42) -> Dict:
        """Time and memory usage for a single LOO experiment."""
        tracemalloc.start()
        t0 = time.perf_counter()
        try:
            self.run_loo_experiment(seed=seed)
        except Exception:
            pass
        elapsed = time.perf_counter() - t0
        _, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return {
            "runtime_seconds": float(elapsed),
            "peak_memory_mb": float(peak_mem / 1024 / 1024),
        }


__all__ = ["LeaveOneOutValidator", "compute_anomaly_consistency", "LOORoundResult"]
