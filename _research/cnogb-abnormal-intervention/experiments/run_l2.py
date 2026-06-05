"""L2 Cross-Asset Analysis: Bond anomaly vs A-share bank sector.

Usage:
    python experiments/run_l2.py --config configs/l2.yaml
    python experiments/run_l2.py --config configs/l2.yaml --dry-run

This script:
  1. Loads bond time-series data from the configured data directory.
  2. Downloads A-share bank stock data via AKShare.
  3. Aligns the two datasets to common trading dates.
  4. Builds anomaly scores for both bond and stock series using the TranAD model.
  5. Runs static and dynamic Granger Causality tests between bond and stock
     anomaly scores.
  6. Computes synchronisation and leading-indicator statistics.
  7. Writes a structured JSON report to results/l2_report.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ── Project root on sys.path ──────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import load_config
from src.data.external import (
    DEFAULT_BANK_TICKERS,
    align_stock_to_bond,
    compute_stock_returns,
    download_multiple_stocks,
)
from src.data.loader import find_paired_files
from src.data.processing import process_pair
from src.data.features import prepare_pair_feature_artifacts
from src.data.sequence import create_sequences
from src.data.dataset import build_dataloaders
from src.evaluate.granger import (
    bidirectional_granger,
    compute_dynamic_granger,
    compute_lag_correlations,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Config helpers ────────────────────────────────────────────────────────────

def _deep_merge(base: dict, override: dict) -> dict:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_l2_config(config_path: str) -> dict:
    config_file = Path(config_path).resolve()
    cfg = load_config(str(config_file))
    base_ref = cfg.get("base_config")
    if base_ref:
        base_path = (config_file.parent / base_ref).resolve()
        base_cfg = load_config(str(base_path))
        cfg = _deep_merge(base_cfg, cfg)
    return cfg


# ── Data loading ──────────────────────────────────────────────────────────────

def load_bond_data(cfg: dict) -> Tuple[Any, str, str]:
    """Return (processed_info, time_col, target_col) for the bond pair."""
    data_cfg = cfg.get("data", {})
    path_cfg = cfg.get("paths", {})
    data_dir = _PROJECT_ROOT / path_cfg.get("raw", data_cfg.get("raw", "data/raw"))

    pairs = find_paired_files(data_dir)
    if not pairs:
        raise FileNotFoundError(f"No data pairs found in: {data_dir}")

    pair_name = data_cfg.get("pair_name")
    pair = next(
        (p for p in pairs if p.get("name") == pair_name),
        pairs[0],
    )
    logger.info("Using bond pair: %s", pair.get("name", "unknown"))

    time_col = data_cfg.get("time_column", "date")
    target_col = data_cfg.get("target_column", "weighted_rate")

    info = process_pair(pair, time_col, target_col)
    if not info.get("valid"):
        raise ValueError(f"Bond pair '{pair_name}' is invalid or has missing columns.")

    return info, time_col, target_col


def load_stock_data(cfg: dict) -> Dict[str, Any]:
    """Download stock data for all configured tickers."""
    data_cfg = cfg.get("data", {})
    stock_cfg = data_cfg.get("stock", {})

    tickers = stock_cfg.get("tickers", DEFAULT_BANK_TICKERS)
    start_date = stock_cfg.get("start_date", "2020-01-01")
    end_date = stock_cfg.get("end_date", "2024-01-01")

    logger.info("Downloading stock data from AKShare (%s → %s)…", start_date, end_date)
    stock_data = download_multiple_stocks(tickers, start_date, end_date)

    if not stock_data:
        raise RuntimeError(
            "No stock data was downloaded. Check network access and AKShare availability."
        )
    return stock_data


def load_l2_data(cfg: dict) -> Tuple[Any, Dict[str, Any], Dict[str, Any]]:
    """Load and align bond + stock data; return (bond_info, stock_data, merged_map).

    ``merged_map`` maps stock code → aligned DataFrame.
    """
    bond_info, time_col, target_col = load_bond_data(cfg)
    stock_data = load_stock_data(cfg)

    data_cfg = cfg.get("data", {})
    stock_cfg = data_cfg.get("stock", {})

    merged_map: Dict[str, Any] = {}
    logger.info("Aligning bond and stock data…")
    for code, stock_df in stock_data.items():
        bond_df_sorted = bond_info["df_train"].copy()
        try:
            merged = align_stock_to_bond(bond_df_sorted, stock_df, date_col=time_col)
            merged_map[code] = merged
            logger.info("  ✓ Aligned with %s: %d common trading days", code, len(merged))
        except ValueError as exc:
            logger.warning("  ✗ Skipping %s: %s", code, exc)

    if not merged_map:
        raise RuntimeError("No overlapping dates found between bond and any stock series.")

    return bond_info, stock_data, merged_map


# ── Anomaly scoring ───────────────────────────────────────────────────────────

def _reconstruction_anomaly_scores(series: np.ndarray, window: int = 16) -> np.ndarray:
    """Simple rolling-reconstruction anomaly score using mean-squared residuals.

    Used as a lightweight proxy when a full TranAD training run is not
    required (e.g. dry-run, or when GPU / data are unavailable).
    """
    scores = np.zeros(len(series))
    for i in range(window, len(series)):
        window_data = series[i - window : i]
        mean_val = window_data.mean()
        scores[i] = (series[i] - mean_val) ** 2
    # Normalise to [0, 1]
    max_score = scores.max()
    if max_score > 0:
        scores = scores / max_score
    return scores


def compute_anomaly_scores(
    series: np.ndarray,
    cfg: dict,
    seed: int = 42,
) -> np.ndarray:
    """Compute anomaly scores for a 1-D time series.

    Attempts a lightweight TranAD-style reconstruction approach; falls back to
    the rolling-residual heuristic if dependencies are unavailable.

    Args:
        series: 1-D array of values.
        cfg: Full experiment configuration dict.
        seed: Random seed for reproducibility.

    Returns:
        1-D array of anomaly scores in [0, 1].
    """
    import torch

    np.random.seed(seed)
    torch.manual_seed(seed)

    training_cfg = cfg.get("training", {})
    window = int(training_cfg.get("sequence_length", 16))

    try:
        from src.models.tranad import TranADConfig, build_tranad
        from src.train.loop import train_epoch, evaluate_model_simple
        from src.data.sequence import create_sequences
        from src.data.dataset import build_dataloaders
        import torch.nn as nn
        import torch.optim as optim

        model_cfg = cfg.get("model", {})
        device = str(training_cfg.get("device", "cpu"))
        num_epochs = int(training_cfg.get("num_epochs", 20))
        lr = float(training_cfg.get("learning_rate", 1e-3))
        batch_size = int(training_cfg.get("batch_size", 32))

        # Normalise series to [0, 1]
        s_min, s_max = series.min(), series.max()
        normed = (series - s_min) / (s_max - s_min + 1e-8)
        normed_2d = normed.reshape(-1, 1)

        X_seq, y_seq = create_sequences(normed_2d, normed_2d, window)
        train_loader, _ = build_dataloaders(X_seq, y_seq, batch_size)

        tranad_cfg = TranADConfig(
            d_model=model_cfg.get("d_model", 128),
            nhead=model_cfg.get("nhead", 4),
            num_layers=model_cfg.get("num_layers", 3),
            dim_feedforward=model_cfg.get("dim_feedforward", 256),
            dropout=model_cfg.get("dropout", 0.1),
            device=device,
            adversarial_weight=model_cfg.get("adversarial_weight", 1.0),
        )
        model = build_tranad(1, config=tranad_cfg, device=device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        for _ in range(num_epochs):
            train_epoch(model, train_loader, criterion, optimizer, device)

        # Compute per-step reconstruction error as anomaly score
        model.eval()
        scores = np.zeros(len(series))
        with torch.no_grad():
            for i in range(window, len(series)):
                x = torch.tensor(
                    normed_2d[i - window : i], dtype=torch.float32
                ).unsqueeze(0).to(device)
                out = model(x)
                if isinstance(out, (list, tuple)):
                    out = out[0]
                err = float(((out - x) ** 2).mean().item())
                scores[i] = err

        max_s = scores.max()
        if max_s > 0:
            scores = scores / max_s
        return scores

    except Exception as exc:
        logger.warning(
            "TranAD anomaly scoring failed (%s); falling back to rolling residual.", exc
        )
        return _reconstruction_anomaly_scores(series, window=window)


# ── Granger analysis ──────────────────────────────────────────────────────────

def run_granger_analysis(
    bond_scores: np.ndarray,
    stock_scores: np.ndarray,
    cfg: dict,
    stock_code: str = "unknown",
) -> Dict:
    """Run full Granger analysis between bond and stock anomaly scores.

    Args:
        bond_scores: Bond anomaly scores (n_samples,).
        stock_scores: Stock anomaly scores (n_samples,).
        cfg: Full experiment configuration dict.
        stock_code: Identifier string used in logging.

    Returns:
        Dict with Granger results and lag-correlation analysis.
    """
    granger_cfg = cfg.get("granger", {})
    max_lag = int(granger_cfg.get("max_lag", 5))
    alpha = float(granger_cfg.get("alpha", 0.05))
    dyn_window = int(granger_cfg.get("dynamic_window_size", 30))
    dyn_stride = int(granger_cfg.get("dynamic_stride", 5))

    logger.info("  Running static Granger causality for %s…", stock_code)
    bidir = bidirectional_granger(bond_scores, stock_scores, max_lag=max_lag, alpha=alpha)

    logger.info("  Running dynamic Granger (window=%d, stride=%d)…", dyn_window, dyn_stride)
    dyn_bond_to_stock = compute_dynamic_granger(
        bond_scores, stock_scores,
        window_size=dyn_window, stride=dyn_stride, max_lag=3, alpha=alpha,
    )
    dyn_stock_to_bond = compute_dynamic_granger(
        stock_scores, bond_scores,
        window_size=dyn_window, stride=dyn_stride, max_lag=3, alpha=alpha,
    )

    logger.info("  Computing lag correlations…")
    lag_corr = compute_lag_correlations(bond_scores, stock_scores, max_lag=max_lag)
    synchronisation = float(np.corrcoef(bond_scores, stock_scores)[0, 1])

    return {
        "stock_code": stock_code,
        "static_granger": {
            "bond_to_stock": bidir["a_to_b"],
            "stock_to_bond": bidir["b_to_a"],
            "direction": bidir["direction"],
        },
        "dynamic_granger": {
            "bond_to_stock": {
                "window_end_indices": dyn_bond_to_stock["window_end_indices"],
                "min_p_values": dyn_bond_to_stock["min_p_values"],
                "is_significant": dyn_bond_to_stock["is_significant"],
                "significant_fraction": dyn_bond_to_stock["significant_fraction"],
            },
            "stock_to_bond": {
                "significant_fraction": dyn_stock_to_bond["significant_fraction"],
            },
        },
        "lag_correlation": lag_corr,
        "synchronisation": synchronisation,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main(config_path: str = "configs/l2.yaml", dry_run: bool = False) -> Dict:
    """Run the L2 cross-asset experiment.

    Args:
        config_path: Path to the YAML configuration file.
        dry_run: If True, skip model training and use synthetic anomaly scores
                 to validate the pipeline end-to-end.

    Returns:
        Report dictionary (also written to ``cfg['output']['report_path']``).
    """
    logger.info("Loading configuration from %s…", config_path)
    cfg = load_l2_config(config_path)

    granger_cfg = cfg.get("granger", {})
    alpha = float(granger_cfg.get("alpha", 0.05))

    # ── 1. Load data ──────────────────────────────────────────────────────────
    logger.info("Loading bond data…")
    bond_info, time_col, target_col = load_bond_data(cfg)
    bond_df = bond_info["df_train"]
    bond_series = bond_df[target_col].astype(float).to_numpy()
    logger.info("  ✓ Bond data: %d samples", len(bond_series))

    if dry_run:
        logger.info("DRY-RUN mode: using synthetic stock data.")
        import pandas as pd

        rng = np.random.default_rng(42)
        stock_codes = [t["code"] for t in DEFAULT_BANK_TICKERS]
        # Create synthetic stock DataFrames matching the bond date range
        dates = pd.to_datetime(bond_df[time_col])
        synthetic_stock_data: Dict[str, Any] = {}
        for code in stock_codes:
            n = len(dates)
            synthetic_stock_data[code] = pd.DataFrame({
                time_col: dates.values,
                "close": rng.random(n) * 10 + 5,
                "volume": rng.integers(1_000_000, 10_000_000, n),
            })
        stock_data = synthetic_stock_data
        merged_map: Dict[str, Any] = {}
        for code, sdf in stock_data.items():
            try:
                merged_map[code] = align_stock_to_bond(bond_df, sdf, date_col=time_col)
            except ValueError:
                pass
    else:
        logger.info("Downloading stock data from AKShare…")
        stock_data = load_stock_data(cfg)
        logger.info("Aligning bond and stock data…")
        merged_map = {}
        for code, stock_df in stock_data.items():
            try:
                merged = align_stock_to_bond(bond_df, stock_df, date_col=time_col)
                merged_map[code] = merged
                logger.info("  ✓ Aligned with %s: %d common trading days", code, len(merged))
            except ValueError as exc:
                logger.warning("  ✗ Skipping %s: %s", code, exc)

    if not merged_map:
        raise RuntimeError("No overlapping dates found between bond and any stock series.")

    # ── 2. Compute anomaly scores and Granger analysis ────────────────────────
    data_cfg = cfg.get("data", {})
    stock_cfg = data_cfg.get("stock", {})
    price_col = stock_cfg.get("price_column", "close")
    eval_cfg = cfg.get("evaluation", {})
    seeds: List[int] = eval_cfg.get("stability_seeds", [42])

    per_stock_results: List[Dict] = []

    for code, merged in merged_map.items():
        logger.info("Processing stock %s…", code)

        # Bond series aligned to this stock's trading dates
        bond_aligned = merged[f"bond_{target_col}"].astype(float).to_numpy()
        stock_aligned = merged[f"stock_{price_col}"].astype(float).to_numpy()

        # Compute anomaly scores (use first seed; stability loop below)
        bond_anomaly = compute_anomaly_scores(bond_aligned, cfg, seed=seeds[0])
        stock_anomaly = compute_anomaly_scores(stock_aligned, cfg, seed=seeds[0])

        granger_result = run_granger_analysis(bond_anomaly, stock_anomaly, cfg, stock_code=code)

        # Stability: repeat across seeds and record fraction of significant results
        stability_results: List[Dict] = []
        for seed in seeds:
            b_sc = compute_anomaly_scores(bond_aligned, cfg, seed=seed)
            s_sc = compute_anomaly_scores(stock_aligned, cfg, seed=seed)
            from src.evaluate.granger import granger_causality_test
            try:
                r = granger_causality_test(b_sc, s_sc, max_lag=3, alpha=alpha)
                stability_results.append({
                    "seed": seed,
                    "conclusion": r["conclusion"],
                    "min_p_value": r["min_p_value"],
                })
            except Exception as exc:
                logger.debug("Stability seed %d failed: %s", seed, exc)

        sig_count = sum(1 for r in stability_results if r["conclusion"] == "significant")
        stability_fraction = sig_count / len(stability_results) if stability_results else 0.0

        per_stock_results.append({
            **granger_result,
            "stability": {
                "seeds": seeds,
                "results": stability_results,
                "significant_fraction": stability_fraction,
            },
        })

    # ── 3. Aggregate results ──────────────────────────────────────────────────
    directions = [r["static_granger"]["direction"] for r in per_stock_results]
    all_sync = [r["synchronisation"] for r in per_stock_results]
    mean_sync = float(np.mean(all_sync)) if all_sync else 0.0

    report: Dict = {
        "experiment": "L2_cross_asset",
        "config_path": str(config_path),
        "bond_samples": int(len(bond_series)),
        "stocks_processed": len(per_stock_results),
        "per_stock": per_stock_results,
        "aggregate": {
            "mean_synchronisation": mean_sync,
            "direction_counts": {
                d: directions.count(d) for d in {"a→b", "b→a", "bidirectional", "none"}
            },
        },
    }

    # ── 4. Write report ───────────────────────────────────────────────────────
    output_cfg = cfg.get("output", {})
    report_path = _PROJECT_ROOT / output_cfg.get("report_path", "results/l2_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False, default=str)
    logger.info("Report written to %s", report_path)

    # ── 5. Print summary ──────────────────────────────────────────────────────
    logger.info("")
    logger.info("═══ L2 Analysis Summary ═══")
    logger.info("Bond samples : %d", report["bond_samples"])
    logger.info("Stocks analysed : %d", report["stocks_processed"])
    logger.info("Mean synchronisation : %.3f", mean_sync)
    for r in per_stock_results:
        code = r["stock_code"]
        direction = r["static_granger"]["direction"]
        sync = r["synchronisation"]
        stab = r["stability"]["significant_fraction"]
        logger.info(
            "  %s  direction=%-14s  sync=%.3f  stability=%.1f%%",
            code, direction, sync, stab * 100,
        )
    logger.info("═══════════════════════════")

    return report


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="L2 Cross-Asset Analysis: Bond anomaly vs A-share bank sector"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/l2.yaml",
        help="Path to the L2 YAML configuration file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Run the full pipeline with synthetic stock data (no AKShare download). "
            "Useful for validating that all modules are importable and the pipeline "
            "runs end-to-end."
        ),
    )
    args = parser.parse_args()
    main(config_path=args.config, dry_run=args.dry_run)
