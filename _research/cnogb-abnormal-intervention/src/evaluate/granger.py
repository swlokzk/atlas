"""Granger Causality analysis for cross-asset anomaly investigation.

Provides static and rolling-window (dynamic) Granger Causality tests
between bond and stock time series, as well as helpers for interpreting
and summarising the results.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def granger_causality_test(
    X: np.ndarray,
    Y: np.ndarray,
    max_lag: int = 5,
    alpha: float = 0.05,
) -> Dict:
    """Test whether X Granger-causes Y.

    Fits a VAR(p) model for each lag order p = 1 … max_lag and records the
    F-test p-value produced by
    ``statsmodels.tsa.stattools.grangercausalitytests``.

    Args:
        X: Potential cause time series, shape (n_samples,).
        Y: Potential effect time series, shape (n_samples,).
        max_lag: Maximum lag order to test.
        alpha: Significance level for the hypothesis test.

    Returns:
        Dict with keys:

        * ``p_values``       – list of F-test p-values for lags 1 … max_lag.
        * ``f_stats``        – list of F-statistics for lags 1 … max_lag.
        * ``significant_lag`` – smallest lag with p < alpha, or None.
        * ``min_p_value``    – minimum p-value across all lags.
        * ``conclusion``     – ``'significant'`` or ``'not_significant'``.

    References:
        Granger, C. W. J. (1969). "Investigating Causal Relations by
        Econometric Models and Cross-spectral Methods." *Econometrica*, 37(3).
    """
    try:
        from statsmodels.tsa.stattools import grangercausalitytests
    except ImportError as exc:
        raise ImportError(
            "statsmodels is required. Install it with: pip install statsmodels"
        ) from exc

    X = np.asarray(X, dtype=float).flatten()
    Y = np.asarray(Y, dtype=float).flatten()

    if len(X) != len(Y):
        raise ValueError(f"X and Y must have the same length. Got {len(X)} vs {len(Y)}.")
    # A VAR(p) model consumes p observations plus needs degrees of freedom for
    # both the restricted and unrestricted regressions; max_lag * 10 is a
    # conservative lower bound that ensures the F-test has meaningful power.
    min_samples = max_lag * 10
    if len(X) < min_samples:
        raise ValueError(
            f"Not enough samples ({len(X)}) for max_lag={max_lag}. "
            f"Need at least {min_samples} samples."
        )

    # grangercausalitytests expects data as [[Y, X]] (Y first)
    data = np.column_stack([Y, X])
    gc_result = grangercausalitytests(data, maxlag=max_lag, verbose=False)

    p_values: List[float] = []
    f_stats: List[float] = []
    for lag in range(1, max_lag + 1):
        # gc_result[lag][0] is a dict of test names → (stat, p-value, df_denom, df_num)
        f_stat, p_val, *_ = gc_result[lag][0]["ssr_ftest"]
        p_values.append(float(p_val))
        f_stats.append(float(f_stat))

    significant_lag: Optional[int] = next(
        (lag for lag, p in enumerate(p_values, start=1) if p < alpha),
        None,
    )

    return {
        "p_values": p_values,
        "f_stats": f_stats,
        "significant_lag": significant_lag,
        "min_p_value": float(min(p_values)),
        "conclusion": "significant" if significant_lag is not None else "not_significant",
    }


def compute_dynamic_granger(
    X: np.ndarray,
    Y: np.ndarray,
    window_size: int = 30,
    stride: int = 5,
    max_lag: int = 3,
    alpha: float = 0.05,
) -> Dict:
    """Compute rolling-window (dynamic) Granger Causality.

    Slides a window of length ``window_size`` over the paired series with
    step ``stride`` and runs ``granger_causality_test`` for each window.
    This reveals whether the causal relationship changes over time.

    Args:
        X: Potential cause time series, shape (n_samples,).
        Y: Potential effect time series, shape (n_samples,).
        window_size: Number of observations per rolling window.
        stride: Step size between consecutive windows.
        max_lag: Maximum lag order passed to each static test.
        alpha: Significance level.

    Returns:
        Dict with keys:

        * ``window_end_indices`` – list of end-of-window sample indices.
        * ``p_values``           – lag-1 p-value for each window.
        * ``min_p_values``       – minimum p-value (across lags) for each window.
        * ``is_significant``     – bool list; True when min_p < alpha.
        * ``significant_fraction`` – share of windows that are significant.
    """
    X = np.asarray(X, dtype=float).flatten()
    Y = np.asarray(Y, dtype=float).flatten()

    n = len(X)
    window_end_indices: List[int] = []
    p_values_lag1: List[float] = []
    min_p_values: List[float] = []

    for start in range(0, n - window_size + 1, stride):
        end = start + window_size
        X_win = X[start:end]
        Y_win = Y[start:end]
        try:
            result = granger_causality_test(X_win, Y_win, max_lag=max_lag, alpha=alpha)
            p_values_lag1.append(result["p_values"][0])
            min_p_values.append(result["min_p_value"])
        except Exception as exc:
            logger.debug("Skipping window [%d:%d]: %s", start, end, exc)
            p_values_lag1.append(float("nan"))
            min_p_values.append(float("nan"))
        window_end_indices.append(end)

    is_significant = [
        (not np.isnan(p)) and (p < alpha) for p in min_p_values
    ]
    valid_count = sum(1 for p in min_p_values if not np.isnan(p))
    significant_fraction = (
        sum(is_significant) / valid_count if valid_count > 0 else 0.0
    )

    return {
        "window_end_indices": window_end_indices,
        "p_values": p_values_lag1,
        "min_p_values": min_p_values,
        "is_significant": is_significant,
        "significant_fraction": float(significant_fraction),
    }


def bidirectional_granger(
    series_a: np.ndarray,
    series_b: np.ndarray,
    max_lag: int = 5,
    alpha: float = 0.05,
) -> Dict:
    """Run Granger tests in both directions between two series.

    Args:
        series_a: First time series (e.g. bond anomaly scores).
        series_b: Second time series (e.g. stock anomaly scores).
        max_lag: Maximum lag order.
        alpha: Significance level.

    Returns:
        Dict with keys:

        * ``a_to_b`` – result of ``granger_causality_test(series_a, series_b)``.
        * ``b_to_a`` – result of ``granger_causality_test(series_b, series_a)``.
        * ``direction`` – ``'a→b'``, ``'b→a'``, ``'bidirectional'``, or ``'none'``.
    """
    a_to_b = granger_causality_test(series_a, series_b, max_lag=max_lag, alpha=alpha)
    b_to_a = granger_causality_test(series_b, series_a, max_lag=max_lag, alpha=alpha)

    a_sig = a_to_b["conclusion"] == "significant"
    b_sig = b_to_a["conclusion"] == "significant"

    if a_sig and b_sig:
        direction = "bidirectional"
    elif a_sig:
        direction = "a→b"
    elif b_sig:
        direction = "b→a"
    else:
        direction = "none"

    return {
        "a_to_b": a_to_b,
        "b_to_a": b_to_a,
        "direction": direction,
    }


def compute_lag_correlations(
    X: np.ndarray,
    Y: np.ndarray,
    max_lag: int = 5,
) -> Dict:
    """Compute cross-correlations at various lags to identify leading indicators.

    Negative lags mean X leads Y; positive lags mean Y leads X.

    Args:
        X: Time series A (n_samples,).
        Y: Time series B (n_samples,).
        max_lag: Maximum absolute lag to evaluate.

    Returns:
        Dict with keys:

        * ``lags``               – list of lag values from -max_lag to +max_lag.
        * ``correlations``       – Pearson correlation at each lag.
        * ``max_correlation_lag`` – lag with the highest absolute correlation.
        * ``max_correlation``    – the correlation value at that lag.
    """
    X = np.asarray(X, dtype=float).flatten()
    Y = np.asarray(Y, dtype=float).flatten()

    lags = list(range(-max_lag, max_lag + 1))
    correlations: List[float] = []

    for lag in lags:
        if lag < 0:
            # Negative lag: X leads Y by abs(lag) periods.
            # X[:-abs(lag)] aligns with Y[abs(lag):] (future Y values).
            abs_lag = abs(lag)
            corr = float(np.corrcoef(X[:-abs_lag], Y[abs_lag:])[0, 1])
        elif lag > 0:
            # Positive lag: Y leads X by lag periods.
            # X[lag:] aligns with Y[:-lag] (past Y values).
            corr = float(np.corrcoef(X[lag:], Y[:-lag])[0, 1])
        else:
            corr = float(np.corrcoef(X, Y)[0, 1])
        correlations.append(corr if not np.isnan(corr) else 0.0)

    max_idx = int(np.argmax(np.abs(correlations)))
    return {
        "lags": lags,
        "correlations": correlations,
        "max_correlation_lag": lags[max_idx],
        "max_correlation": correlations[max_idx],
    }


__all__ = [
    "bidirectional_granger",
    "compute_dynamic_granger",
    "compute_lag_correlations",
    "granger_causality_test",
]
