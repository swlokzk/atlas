from typing import Any, Dict, Iterable, Optional

from .preprocessing import load_and_clean_file, sort_by_time


def _detect_pair_type(pair_name: str) -> str:
    return "fc7" if "fc7" in pair_name.lower() else "original"


def _select_feature_columns(
    df_train_columns: list[str],
    df_exam_columns: list[str],
    time_col: str,
    target_col: str,
    pair_type: str,
    derived_feature_columns: Optional[Iterable[str]] = None,
) -> list[str]:
    shared_columns = [column for column in df_train_columns if column in df_exam_columns]
    if pair_type == "fc7" and derived_feature_columns:
        return [column for column in derived_feature_columns if column in shared_columns]
    return [column for column in shared_columns if column not in {time_col, target_col}]


def process_and_validate_pair(
    pair: Dict[str, str],
    time_col: str,
    target_col: str,
    derived_feature_columns: Optional[Iterable[str]] = None,
    categorical_columns: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Load, clean, validate, and sort a train/exam pair."""
    pair_name = pair.get("name", "")
    info: Dict[str, Any] = {"valid": False, "pair_name": pair_name, "pair_type": _detect_pair_type(pair_name)}
    if not pair.get("train") or not pair.get("exam"):
        return info

    df_train, train_cleaned_columns = load_and_clean_file(pair["train"])
    df_exam, exam_cleaned_columns = load_and_clean_file(pair["exam"])

    if time_col not in df_train.columns or time_col not in df_exam.columns:
        return info
    if target_col not in df_train.columns or target_col not in df_exam.columns:
        return info

    df_train_sorted = sort_by_time(df_train, time_col)
    df_exam_sorted = sort_by_time(df_exam, time_col)

    feature_columns = _select_feature_columns(
        train_cleaned_columns,
        exam_cleaned_columns,
        time_col,
        target_col,
        info["pair_type"],
        derived_feature_columns=derived_feature_columns,
    )
    if not feature_columns:
        return info

    categorical_features = [column for column in (categorical_columns or []) if column in feature_columns]
    numerical_features = [column for column in feature_columns if column not in categorical_features]

    info.update({
        "valid": True,
        "df_train_cleaned": df_train,
        "df_exam_cleaned": df_exam,
        "df_train_sorted": df_train_sorted,
        "df_exam_sorted": df_exam_sorted,
        "df_train": df_train_sorted,
        "df_exam": df_exam_sorted,
        "feature_columns": feature_columns,
        "categorical_features": categorical_features,
        "numerical_features": numerical_features,
        "time_feature": time_col,
        "target_column": target_col,
    })
    return info


def process_pair(
    pair: Dict[str, str],
    time_col: str,
    target_col: str,
    derived_feature_columns: Optional[Iterable[str]] = None,
    categorical_columns: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    return process_and_validate_pair(
        pair,
        time_col,
        target_col,
        derived_feature_columns=derived_feature_columns,
        categorical_columns=categorical_columns,
    )
