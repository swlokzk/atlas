"""External data integration: download A-share stock data and align with bond data."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# A-share bank tickers that are the primary institutional holders of policy financial bonds.
DEFAULT_BANK_TICKERS: List[Dict[str, str]] = [
    {"code": "601398", "name": "工商银行"},
    {"code": "601288", "name": "农业银行"},
    {"code": "601988", "name": "中国银行"},
    {"code": "601939", "name": "建设银行"},
    {"code": "600036", "name": "招商银行"},
]


def download_ak_stock_data(
    stock_code: str,
    start_date: str,
    end_date: str,
    adjust: str = "qfq",
) -> pd.DataFrame:
    """Download daily A-share stock data from AKShare.

    Args:
        stock_code: A-share stock code (e.g. '601398' for ICBC).
        start_date: Start date in 'YYYY-MM-DD' format.
        end_date: End date in 'YYYY-MM-DD' format.
    adjust: Price adjustment method. 'qfq' = forward-adjusted (前复权).

    Returns:
        DataFrame with columns: date, open, close, high, low, volume, amount, turnover_rate.

    Raises:
        ImportError: If akshare is not installed.
        ValueError: If the downloaded data is empty or malformed.
    """
    try:
        import akshare as ak
    except ImportError as exc:
        raise ImportError("akshare is required. Install it with: pip install akshare") from exc

    logger.info("Downloading stock data: %s (%s to %s)", stock_code, start_date, end_date)
    df = ak.stock_zh_a_hist(
        symbol=stock_code,
        period="daily",
        start_date=start_date.replace("-", ""),
        end_date=end_date.replace("-", ""),
        adjust=adjust,
    )

    if df is None or df.empty:
        raise ValueError(f"No data returned for stock_code={stock_code} between {start_date} and {end_date}")

    # Normalise column names produced by AKShare (may vary by version)
    col_map = {
        "日期": "date",
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
        "成交额": "amount",
        "换手率": "turnover_rate",
        # English fallback names (some AKShare versions)
        "date": "date",
        "open": "open",
        "close": "close",
        "high": "high",
        "low": "low",
        "volume": "volume",
        "amount": "amount",
        "turnover_rate": "turnover_rate",
    }
    df = df.rename(columns={c: col_map[c] for c in df.columns if c in col_map})

    if "date" not in df.columns:
        # Attempt positional rename for unexpected column layouts
        expected = ["date", "open", "close", "high", "low", "volume", "amount", "turnover_rate"]
        if len(df.columns) >= len(expected):
            logger.warning(
                "Unrecognised AKShare columns %s; applying positional rename to %s.",
                df.columns.tolist(),
                expected[: len(df.columns)],
            )
            df.columns = expected[: len(df.columns)]
        else:
            raise ValueError(f"Cannot parse columns from AKShare response: {df.columns.tolist()}")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    logger.info("Downloaded %d rows for %s", len(df), stock_code)
    return df


def download_multiple_stocks(
    tickers: Optional[List[Dict[str, str]]] = None,
    start_date: str = "2020-01-01",
    end_date: str = "2024-01-01",
) -> Dict[str, pd.DataFrame]:
    """Download data for multiple A-share stocks.

    Args:
        tickers: List of dicts with 'code' and 'name' keys.  Defaults to
            ``DEFAULT_BANK_TICKERS``.
        start_date: Start date in 'YYYY-MM-DD' format.
        end_date: End date in 'YYYY-MM-DD' format.

    Returns:
        Dict mapping stock code → DataFrame.  Failed downloads are logged and
        omitted from the result rather than raising.
    """
    if tickers is None:
        tickers = DEFAULT_BANK_TICKERS

    stock_data: Dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        code = ticker["code"]
        name = ticker.get("name", code)
        try:
            df = download_ak_stock_data(code, start_date, end_date)
            stock_data[code] = df
            logger.info("  ✓ %s (%s): %d samples", name, code, len(df))
        except Exception as exc:
            logger.warning("  ✗ Failed to download %s (%s): %s", name, code, exc)

    return stock_data


def align_stock_to_bond(
    bond_df: pd.DataFrame,
    stock_df: pd.DataFrame,
    date_col: str = "date",
) -> pd.DataFrame:
    """Align bond and stock data to their common trading dates (inner join).

    Args:
        bond_df: Bond time-series DataFrame; must contain a date column.
        stock_df: Stock time-series DataFrame; must contain a date column.
        date_col: Name of the date column in both DataFrames.

    Returns:
        Merged DataFrame where non-date bond columns are prefixed with
        ``bond_`` and non-date stock columns are prefixed with ``stock_``.

    Raises:
        ValueError: If either input is empty or the merged result is empty.
    """
    if bond_df.empty:
        raise ValueError("bond_df is empty")
    if stock_df.empty:
        raise ValueError("stock_df is empty")

    bond = bond_df.copy()
    stock = stock_df.copy()

    bond[date_col] = pd.to_datetime(bond[date_col])
    stock[date_col] = pd.to_datetime(stock[date_col])

    # Prefix non-date columns
    bond = bond.rename(columns={c: f"bond_{c}" for c in bond.columns if c != date_col})
    stock = stock.rename(columns={c: f"stock_{c}" for c in stock.columns if c != date_col})

    merged = pd.merge(bond, stock, on=date_col, how="inner")
    merged = merged.sort_values(date_col).reset_index(drop=True)

    if merged.empty:
        raise ValueError(
            "No common trading dates found between bond_df and stock_df. "
            f"Bond range: {bond_df[date_col].min()} – {bond_df[date_col].max()}, "
            f"Stock range: {stock_df[date_col].min()} – {stock_df[date_col].max()}"
        )

    logger.info(
        "Aligned data: bond=%d rows, stock=%d rows → merged=%d common trading days",
        len(bond_df),
        len(stock_df),
        len(merged),
    )
    return merged


def compute_stock_returns(df: pd.DataFrame, close_col: str = "close") -> pd.Series:
    """Compute log returns from a close-price series.

    Args:
        df: DataFrame containing a close price column.
        close_col: Name of the close price column.

    Returns:
        Series of log returns (same index, first value is NaN).
    """
    import numpy as np

    prices = df[close_col].astype(float)
    returns = np.log(prices / prices.shift(1))
    return returns


__all__ = [
    "DEFAULT_BANK_TICKERS",
    "align_stock_to_bond",
    "compute_stock_returns",
    "download_ak_stock_data",
    "download_multiple_stocks",
]
