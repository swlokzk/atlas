from pathlib import Path
import re
from typing import Tuple

import pandas as pd


def clean_column_names_general(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cleaned_columns = []
    for column_name in df.columns:
        cleaned_column_name = str(column_name).replace("\n", "").strip()
        cleaned_column_name = re.sub(r"^因子\d+[_\s]*", "", cleaned_column_name)
        cleaned_column_name = (
            cleaned_column_name.replace(":", "_")
            .replace("-", "")
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("（", "")
            .replace("）", "")
            .replace("%", "")
            .replace("，", "_")
        )
        cleaned_column_name = re.sub(r"_+", "_", cleaned_column_name).strip("_").strip()
        cleaned_columns.append(cleaned_column_name)

    df.columns = cleaned_columns
    return df


def load_and_clean_file(path: str, **read_kwargs) -> Tuple[pd.DataFrame, list[str]]:
    df = pd.read_excel(path, **read_kwargs)
    df = clean_column_names_general(df)
    return df, df.columns.tolist()


def sort_by_time(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    if time_col not in df.columns:
        return df.copy()
    df = df.copy()
    numeric_sort_key = pd.to_numeric(df[time_col], errors="coerce")
    if numeric_sort_key.notna().any() and numeric_sort_key.notna().sum() >= max(1, len(df) // 2):
        sort_key = numeric_sort_key
    else:
        sort_key = pd.to_datetime(df[time_col], errors="coerce")
    if getattr(sort_key, "isna", None) is not None and sort_key.isna().all():
        sort_key = df[time_col]
    df = df.assign(_sort_key=sort_key).sort_values("_sort_key").drop(columns=["_sort_key"]).reset_index(drop=True)
    return df
