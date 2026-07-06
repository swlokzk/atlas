"""Feature engineering helpers for microstructure, scaling, and notebook-style preprocessing."""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


def safe_divide(numerator, denominator):
    numerator_array = np.asarray(numerator, dtype=float)
    denominator_array = np.asarray(denominator, dtype=float)
    result = np.zeros_like(numerator_array, dtype=float)
    np.divide(numerator_array, denominator_array, out=result, where=denominator_array != 0)
    return result


def compute_bid_ask_spread(frame: pd.DataFrame, bid_col: str = "bid", ask_col: str = "ask") -> pd.Series:
    if bid_col not in frame.columns or ask_col not in frame.columns:
        return pd.Series(index=frame.index, dtype=float)
    return frame[ask_col] - frame[bid_col]


def rolling_std(series: pd.Series, window: int = 20) -> pd.Series:
    return series.rolling(window=window, min_periods=1).std()


def autocorr_lag(series: pd.Series, lag: int = 1) -> pd.Series:
    shifted = series.shift(lag)
    return series.rolling(window=lag + 1, min_periods=lag + 1).corr(shifted)


def price_impact(price: pd.Series, volume: pd.Series) -> pd.Series:
    return pd.Series(safe_divide(price.diff().fillna(0.0), volume.replace(0, np.nan).fillna(0.0)), index=price.index)


def kyles_lambda(price: pd.Series, volume: pd.Series, window: int = 20) -> pd.Series:
    delta_price = price.diff()
    delta_volume = volume.diff()
    covariance = delta_price.rolling(window=window, min_periods=window).cov(delta_volume)
    variance = delta_volume.rolling(window=window, min_periods=window).var()
    return covariance / variance.replace(0, np.nan)


def buy_sell_imbalance(buy_volume: pd.Series, sell_volume: pd.Series) -> pd.Series:
    numerator = buy_volume - sell_volume
    denominator = buy_volume + sell_volume
    return pd.Series(safe_divide(numerator, denominator), index=buy_volume.index)


def realized_volatility(returns: pd.Series, window: int = 20) -> pd.Series:
    squared = returns.pow(2)
    return squared.rolling(window=window, min_periods=1).sum().pow(0.5)


def yield_curve_slope(long_yield: pd.Series, short_yield: pd.Series) -> pd.Series:
    return long_yield - short_yield


def build_feature_frame(frame: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    features = frame.copy()

    if {"ask", "bid"}.issubset(features.columns):
        features["bid_ask_spread"] = compute_bid_ask_spread(features)
        features["spread_volatility"] = rolling_std(features["bid_ask_spread"], window=window)

    if "price" in features.columns and "volume" in features.columns:
        features["price_impact"] = price_impact(features["price"], features["volume"])
        features["kyles_lambda"] = kyles_lambda(features["price"], features["volume"], window=window)

    if {"buy_volume", "sell_volume"}.issubset(features.columns):
        features["buy_sell_imbalance"] = buy_sell_imbalance(features["buy_volume"], features["sell_volume"])

    if "return" in features.columns:
        features["realized_volatility"] = realized_volatility(features["return"], window=window)
        features["return_autocorr"] = autocorr_lag(features["return"], lag=1)

    if {"yield_long", "yield_short"}.issubset(features.columns):
        features["yield_curve_slope"] = yield_curve_slope(features["yield_long"], features["yield_short"])

    return features


def fit_feature_preprocessor(
    feature_frame: pd.DataFrame,
    numeric_columns: Iterable[str],
    categorical_columns: Iterable[str],
) -> Optional[ColumnTransformer]:
    transformers = []
    numeric_columns = list(numeric_columns)
    categorical_columns = list(categorical_columns)

    if numeric_columns:
        transformers.append(("num", MinMaxScaler(), numeric_columns))
    if categorical_columns:
        try:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
        transformers.append(("cat", encoder, categorical_columns))

    if not transformers:
        return None

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    preprocessor.fit(feature_frame)
    return preprocessor


def transform_features(preprocessor: Optional[ColumnTransformer], feature_frame: pd.DataFrame) -> np.ndarray:
    if preprocessor is None:
        return feature_frame.to_numpy()
    return preprocessor.transform(feature_frame)


def fit_target_scaler(target_values: pd.DataFrame | pd.Series | np.ndarray) -> MinMaxScaler:
    scaler = MinMaxScaler()
    target_frame = target_values.to_frame() if isinstance(target_values, pd.Series) else target_values
    scaler.fit(target_frame)
    return scaler


def transform_target(scaler: MinMaxScaler, target_values: pd.DataFrame | pd.Series | np.ndarray) -> np.ndarray:
    target_frame = target_values.to_frame() if isinstance(target_values, pd.Series) else target_values
    return scaler.transform(target_frame)


def prepare_pair_feature_artifacts(processed_info: dict, training_params: dict) -> dict:
    feature_columns = processed_info.get("feature_columns", [])
    categorical_features = processed_info.get("categorical_features", [])
    numerical_features = processed_info.get("numerical_features", [column for column in feature_columns if column not in categorical_features])
    target_column = processed_info.get("target_column")

    df_train = processed_info.get("df_train_sorted")
    if df_train is None:
        df_train = processed_info.get("df_train")
    df_exam = processed_info.get("df_exam_sorted")
    if df_exam is None:
        df_exam = processed_info.get("df_exam")
    if df_train is None or df_exam is None or target_column is None:
        raise ValueError("processed_info is missing train/exam frames or target_column")

    X_train_frame = df_train[feature_columns]
    X_exam_frame = df_exam[feature_columns]

    feature_preprocessor = fit_feature_preprocessor(X_train_frame, numerical_features, categorical_features)
    X_train_transformed_features = transform_features(feature_preprocessor, X_train_frame)
    X_exam_transformed_features = transform_features(feature_preprocessor, X_exam_frame)

    target_scaler = fit_target_scaler(df_train[[target_column]])
    y_train_scaled = transform_target(target_scaler, df_train[[target_column]])
    y_exam_scaled = transform_target(target_scaler, df_exam[[target_column]])

    processed_info = dict(processed_info)
    processed_info.update(
        {
            "X_train_transformed_features": X_train_transformed_features,
            "X_exam_transformed_features": X_exam_transformed_features,
            "y_train_scaled": y_train_scaled,
            "y_exam_scaled": y_exam_scaled,
            "feature_preprocessor": feature_preprocessor,
            "target_scaler": target_scaler,
            "feature_count_transformed": int(X_train_transformed_features.shape[-1]),
            "sequence_length": training_params.get("sequence_length") or training_params.get("SEQUENCE_LENGTH"),
            "batch_size": training_params.get("batch_size") or training_params.get("BATCH_SIZE"),
        }
    )
    return processed_info


__all__ = [
    "autocorr_lag",
    "build_feature_frame",
    "buy_sell_imbalance",
    "compute_bid_ask_spread",
    "fit_feature_preprocessor",
    "fit_target_scaler",
    "kyles_lambda",
    "price_impact",
    "prepare_pair_feature_artifacts",
    "realized_volatility",
    "rolling_std",
    "safe_divide",
    "transform_features",
    "transform_target",
    "yield_curve_slope",
]
