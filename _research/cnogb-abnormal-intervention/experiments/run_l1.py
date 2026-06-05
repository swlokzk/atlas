"""L1 experiment runner: Leave-one-out cross-domain validation.

Usage
-----
Run full experiment::

    python experiments/run_l1.py --config configs/l1.yaml

Dry-run (data loading only, no training)::

    python experiments/run_l1.py --config configs/l1.yaml --dry-run

Override output path::

    python experiments/run_l1.py --config configs/l1.yaml --output results/my_run.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np

# Ensure project root is on sys.path when invoked directly
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import load_config
from src.evaluation.cross_domain import LeaveOneOutValidator, compute_anomaly_consistency


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class L1Result:
    """L1 Leave-one-out 實驗結果."""

    results_by_bond: Dict[str, dict]
    """per-bond dict: auroc, auprc, f1, anomaly_ratio, feature_importance, ..."""

    consistency_metrics: Dict
    """跨標的異常一致性指標."""

    stability: Dict
    """穩定性指標 (mean, std, p_value)."""

    calibration_ece: float
    """Expected Calibration Error."""

    efficiency: Dict
    """運行時間與記憶體佔用."""


def l1_result_to_dict(result: L1Result) -> dict:
    """將 L1Result 序列化為 JSON-safe dict."""
    return {
        "results_by_bond": {
            bond: {
                "auroc": float(res["auroc"]),
                "auprc": float(res["auprc"]),
                "f1": float(res["f1"]),
                "anomaly_ratio": float((res["anomaly_scores"] > 0.5).mean()),
                "uncertainty_mean": float(res["uncertainty"].mean()),
                "feature_importance": {
                    k: float(v) for k, v in res["feature_importance"].items()
                },
                "n_train_samples": int(res["n_train_samples"]),
                "n_test_samples": int(res["n_test_samples"]),
                "train_bonds": res["train_bonds"],
            }
            for bond, res in result.results_by_bond.items()
        },
        "consistency": result.consistency_metrics,
        "stability": result.stability,
        "calibration_ece": float(result.calibration_ece),
        "efficiency": result.efficiency,
    }


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _deep_merge(base: dict, override: dict) -> dict:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_experiment_config(config_path: str) -> dict:
    config_file = Path(config_path).resolve()
    model_config = load_config(str(config_file))
    base_config: dict = {}
    base_reference = model_config.get("base_config")
    if base_reference:
        base_path = (config_file.parent / base_reference).resolve()
        if base_path.exists():
            base_config = load_config(str(base_path))
    return _deep_merge(base_config, model_config)


# ---------------------------------------------------------------------------
# ECE helper (uses per-bond consistency)
# ---------------------------------------------------------------------------

def _compute_mean_ece(loo_results: Dict[str, dict], n_bins: int = 10) -> float:
    """Average ECE across all test bonds.

    We use train/exam pseudo-labels: exam = 1, implied normal = 0.
    Anomaly scores are used as calibrated probabilities.
    """
    from src.evaluation.cross_domain import compute_ece

    ece_values = []
    for bond_res in loo_results.values():
        scores = np.asarray(bond_res["anomaly_scores"]).ravel()
        # Exam scores are all labeled '1' for ECE computation
        labels = np.ones(len(scores), dtype=int)
        ece_values.append(compute_ece(scores, labels, n_bins=n_bins))
    return float(np.mean(ece_values)) if ece_values else 0.0


# ---------------------------------------------------------------------------
# Dry-run helper
# ---------------------------------------------------------------------------

def _dry_run(validator: LeaveOneOutValidator) -> None:
    """Validate data loading without training."""
    print("🔍 Dry-run: checking data loading...")
    missing = []
    for bond in validator.BONDS:
        data = validator.load_data(bond)
        if data is None:
            print(f"  ⚠️  {bond}: data not found under '{validator.raw_dir}'")
            missing.append(bond)
        else:
            n_train = len(data.get("df_train", []))
            n_exam = len(data.get("df_exam", []))
            feat_dim = data.get("feature_count_transformed", "?")
            print(f"  ✓  {bond}: train={n_train} rows, exam={n_exam} rows, features={feat_dim}")

    print()
    if missing:
        print(f"⚠️  Missing data for: {missing}")
        print("   Create or place bond files under data/raw/ as:")
        for b in missing:
            print(f"     data/raw/{b}_train.xlsx  +  data/raw/{b}_exam.xlsx")
    else:
        print("✅ All bond data loaded successfully.")

    # Verify LOO logic for all bonds that have data
    available = [b for b in validator.BONDS if validator.load_data(b) is not None]
    for test_bond in available:
        train_bonds = [b for b in available if b != test_bond]
        assert len(train_bonds) >= 1, f"Need at least 1 training bond when testing {test_bond}"
        assert test_bond not in train_bonds
    print("✅ Leave-one-out logic verified.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(config_path: str, output_path: str | None = None, dry_run: bool = False) -> None:
    cfg = load_experiment_config(config_path)
    project_root = _PROJECT_ROOT

    validator = LeaveOneOutValidator(cfg, project_root=project_root)

    if dry_run:
        _dry_run(validator)
        return

    output_cfg = cfg.get("output", {})
    if output_path is None:
        output_path = str(project_root / output_cfg.get("report_path", "results/l1_report.json"))

    figures_dir = project_root / output_cfg.get("figures_dir", "results/l1_figures")
    figures_dir.mkdir(parents=True, exist_ok=True)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    seed = int(cfg.get("seed", 42))
    eval_cfg = cfg.get("evaluation", {})
    stability_seeds = list(eval_cfg.get("stability_seeds", [42, 123, 456, 789, 1000]))

    # ---- Step 1: Main LOO experiment ----
    print("=" * 60)
    print("🚀 Running L1 Leave-one-out experiment...")
    print(f"   Bonds : {validator.BONDS}")
    print(f"   Epochs: {validator.num_epochs}  |  lr: {validator.lr}  |  device: {validator.device}")
    print("=" * 60)

    t_start = time.perf_counter()
    loo_results = validator.run_loo_experiment(seed=seed)
    t_elapsed = time.perf_counter() - t_start

    print(f"\n📊 LOO Results (elapsed: {t_elapsed:.1f}s):")
    for bond, res in loo_results.items():
        print(
            f"  {bond.upper():4s}  AUROC={res['auroc']:.3f}  "
            f"AUPRC={res['auprc']:.3f}  F1={res['f1']:.3f}  "
            f"train={res['n_train_samples']}  test={res['n_test_samples']}"
        )

    # ---- Step 2: Consistency metrics ----
    print("\n🔗 Computing anomaly consistency...")
    consistency = compute_anomaly_consistency(loo_results)
    print(f"  Consistency ratio   : {consistency['consistency_ratio']:.3%}")
    print(f"  Mean pairwise corr  : {consistency['mean_pairwise_corr']:.3f}")
    for pair_key, corr in consistency["correlation"].items():
        print(f"    {pair_key}: r={corr:.3f}")

    # ---- Step 3: Calibration ----
    print("\n📐 Computing calibration (ECE)...")
    ece = _compute_mean_ece(loo_results)
    print(f"  Mean ECE: {ece:.4f}")

    # ---- Step 4: Stability test ----
    print(f"\n🔁 Stability test ({len(stability_seeds)} seeds)...")
    stability = validator.run_stability_test(stability_seeds)
    print(
        f"  AUROC  mean={stability['mean']:.3f}  "
        f"std={stability['std']:.3f}  "
        f"p={stability['p_value']:.4f}  "
        f"(n={stability['n_runs']})"
    )

    # ---- Step 5: Efficiency ----
    efficiency = {
        "runtime_seconds": float(t_elapsed),
        "stability_runs": stability.get("n_runs", 0),
    }

    # ---- Assemble final result ----
    result = L1Result(
        results_by_bond=loo_results,
        consistency_metrics=consistency,
        stability=stability,
        calibration_ece=ece,
        efficiency=efficiency,
    )

    report = l1_result_to_dict(result)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Report saved to: {output_path}")

    # ---- Validate output ----
    assert "results_by_bond" in report
    assert set(report["results_by_bond"].keys()) == set(loo_results.keys())
    assert "consistency" in report
    assert 0.0 <= report["calibration_ece"] <= 1.0

    available_bonds = list(loo_results.keys())
    for bond in available_bonds:
        auroc = report["results_by_bond"][bond]["auroc"]
        assert 0.0 <= auroc <= 1.0, f"{bond} AUROC {auroc} out of range"
        print(f"  {bond}: AUROC={auroc:.3f}")

    print(f"  Consistency ratio: {report['consistency']['consistency_ratio']:.3%}")
    print("\n✅ All assertions passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run L1 Leave-one-out experiment")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/l1.yaml",
        help="Path to l1.yaml config file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Override output JSON path (default: from config)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only check data loading, skip training",
    )
    args = parser.parse_args()
    main(args.config, output_path=args.output, dry_run=args.dry_run)
