from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from utils import safe_drop_columns


def validate_input(df: pd.DataFrame, target_col: str) -> None:
    if df.empty:
        raise ValueError("Ulazni DataFrame je prazan.")
    if target_col not in df.columns:
        raise ValueError(f"Target kolona '{target_col}' ne postoji.")


def normalize_text_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    result = df.copy()
    object_cols = result.select_dtypes(include=["object"]).columns
    normalized_cells = 0
    for col in object_cols:
        before = result[col].copy()
        result[col] = result[col].astype(str).str.strip().str.lower()
        result[col] = result[col].replace({"": np.nan, "nan": np.nan, "none": np.nan})
        normalized_cells += int((before.notna() & (before != result[col])).sum())
    return result, normalized_cells


def standardize_known_categories(df: pd.DataFrame, value_mappings: dict[str, str]) -> tuple[pd.DataFrame, int]:
    result = df.copy()
    object_cols = result.select_dtypes(include=["object"]).columns
    changes = 0
    for col in object_cols:
        before = result[col].copy()
        result[col] = result[col].replace(value_mappings)
        changes += int((before.notna() & (before != result[col])).sum())
    return result, changes


def convert_numeric_strings(df: pd.DataFrame, numeric_candidates: list[str] | None = None) -> tuple[pd.DataFrame, list[str]]:
    result = df.copy()
    candidates = numeric_candidates or list(result.columns)
    converted_columns: list[str] = []
    for col in candidates:
        if col not in result.columns or result[col].dtype != "object":
            continue
        converted = pd.to_numeric(result[col], errors="coerce")
        ratio = converted.notna().mean()
        if ratio >= 0.8:
            result[col] = converted
            converted_columns.append(col)
    return result, converted_columns



def remove_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    before = len(df)
    result = df.drop_duplicates()
    return result, int(before - len(result))


def build_cleaning_report(before_df: pd.DataFrame, after_df: pd.DataFrame, meta: dict[str, Any]) -> dict[str, Any]:
    return {
        "missing_before": int(before_df.isna().sum().sum()),
        "missing_after": int(after_df.isna().sum().sum()),
        "duplicates_before": int(meta.get("duplicates_before", 0)),
        "standardized_values": int(meta.get("standardized_values", 0)),
        "outlier_counts": meta.get("outlier_counts", {}),
        "normalized_values": int(meta.get("normalized_values", 0)),
        "type_converted_columns": meta.get("type_converted_columns", []),
    }


def prepare_base_dataset(df: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    target = config["target_column"]
    validate_input(df, target)
    result = safe_drop_columns(df.copy(), config.get("drop_columns", []))
    result, normalized_values = normalize_text_columns(result)
    result, standardized_values = standardize_known_categories(result, config.get("value_mappings", {}))
    result, type_converted_columns = convert_numeric_strings(result, numeric_candidates=[c for c in result.columns if c != target])
    duplicates_before = int(result.duplicated().sum())
    result, _ = remove_duplicates(result)
    meta = {
        "duplicates_before": duplicates_before,
        "standardized_values": standardized_values,
        "normalized_values": normalized_values,
        "type_converted_columns": type_converted_columns,
        "drop_columns": [c for c in config.get("drop_columns", []) if c in df.columns],
    }
    return result, meta


def fit_cleaning_artifacts(train_df: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    num_strategy = config.get("numeric_impute_strategy", "median")
    cat_strategy = config.get("categorical_impute_strategy", "mode_or_unknown")
    multiplier = config.get("iqr_multiplier", 1.5)
    outlier_method = config.get("outlier_method", "iqr_clip")

    num_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = train_df.select_dtypes(exclude=[np.number]).columns.tolist()

    fill_values: dict[str, Any] = {}
    for col in num_cols:
        fill_values[col] = train_df[col].median() if num_strategy == "median" else train_df[col].mean()
    for col in cat_cols:
        if cat_strategy == "mode_or_unknown":
            modes = train_df[col].mode(dropna=True)
            fill_values[col] = modes.iloc[0] if not modes.empty else "unknown"

    outlier_bounds: dict[str, tuple[float, float]] = {}
    if outlier_method != "none":
        for col in num_cols:
            q1, q3 = train_df[col].quantile(0.25), train_df[col].quantile(0.75)
            iqr = q3 - q1
            if iqr == 0 or pd.isna(iqr):
                continue
            outlier_bounds[col] = (float(q1 - multiplier * iqr), float(q3 + multiplier * iqr))

    return {
        "fill_values": fill_values,
        "outlier_bounds": outlier_bounds,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
    }


def apply_cleaning_artifacts(df: pd.DataFrame, artifacts: dict[str, Any], outlier_method: str = "iqr_clip") -> tuple[pd.DataFrame, dict[str, Any]]:
    result = df.copy()
    missing_fixed_by_column: dict[str, int] = {}
    for col, fill_value in artifacts.get("fill_values", {}).items():
        if col not in result.columns:
            continue
        missing_count = int(result[col].isna().sum())
        if missing_count > 0:
            result[col] = result[col].fillna(fill_value)
        missing_fixed_by_column[col] = missing_count

    outlier_counts: dict[str, int] = {}
    if outlier_method != "none":
        for col, bounds in artifacts.get("outlier_bounds", {}).items():
            if col not in result.columns:
                continue
            lower, upper = bounds
            mask = (result[col] < lower) | (result[col] > upper)
            outlier_counts[col] = int(mask.sum())
            result[col] = result[col].clip(lower=lower, upper=upper)

    return result, {
        "missing_fixed_by_column": missing_fixed_by_column,
        "outlier_counts": outlier_counts,
    }
