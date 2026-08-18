from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from cleaning import apply_cleaning_artifacts, build_cleaning_report, fit_cleaning_artifacts, prepare_base_dataset
from preprocessing import (
    build_model_pipeline,
    build_preprocessor,
    fit_and_predict,
    infer_feature_types,
    split_features_target,
    split_train_test,
)


def _prepare_target(y: pd.Series, mapping: dict[Any, int] | None = None) -> pd.Series:
    if mapping:
        lowered_mapping = {str(k).lower(): v for k, v in mapping.items()}
        mapped = y.astype(str).str.lower().map(lowered_mapping)
        if mapped.notna().all():
            return mapped.astype(int)
    if y.dtype.kind in "biu":
        return y.astype(int)
    uniques = sorted(y.dropna().astype(str).str.lower().unique())
    auto_map = {label: idx for idx, label in enumerate(uniques)}
    return y.astype(str).str.lower().map(auto_map).astype(int)


def _with_target(X: pd.DataFrame, y: pd.Series, target_col: str) -> pd.DataFrame:
    result = X.copy()
    result[target_col] = y.values
    return result


def evaluate_classification(y_true: pd.Series, y_pred: pd.Series) -> dict[str, Any]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def train_and_evaluate_split(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    config: dict[str, Any],
) -> dict[str, Any]:
    train_mask = X_train.notna().all(axis=1)
    X_train = X_train.loc[train_mask]
    y_train = y_train.loc[train_mask]
    num_cols, cat_cols = infer_feature_types(X_train)
    preprocessor = build_preprocessor(num_cols, cat_cols)
    model_pipeline = build_model_pipeline(preprocessor, config["random_state"])
    preds = fit_and_predict(model_pipeline, X_train, y_train, X_test)
    metrics = evaluate_classification(y_test, preds)
    metrics["n_train"] = int(len(X_train))
    metrics["n_test"] = int(len(X_test))
    return metrics


def run_pipeline(df: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    target_col = config["target_column"]
    base_df, base_meta = prepare_base_dataset(df, config)
    X, y = split_features_target(base_df, target_col)
    y = _prepare_target(y, config.get("target_mappings"))
    X_train, X_test, y_train, y_test = split_train_test(
        X, y, test_size=config["test_size"], random_state=config["random_state"], stratify=True
    )
    complete_test = X_test.notna().all(axis=1)
    if not bool(complete_test.any()):
        raise ValueError(
            "Nema test redova bez praznina; baseline i cleaned se ne mogu usporediti na istom skupu."
        )

    artifacts = fit_cleaning_artifacts(X_train, config)
    outlier_method = config.get("outlier_method", "iqr_clip")
    cleaned_X_train, train_clean_meta = apply_cleaning_artifacts(X_train, artifacts, outlier_method)
    cleaned_X_test, test_clean_meta = apply_cleaning_artifacts(X_test, artifacts, outlier_method)

    y_test_eval = y_test.loc[complete_test]
    X_test_eval = X_test.loc[complete_test]
    baseline_metrics = train_and_evaluate_split(X_train, X_test_eval, y_train, y_test_eval, config)
    cleaned_metrics = train_and_evaluate_split(
        cleaned_X_train, cleaned_X_test.loc[complete_test], y_train, y_test_eval, config
    )

    outlier_counts: dict[str, int] = {}
    missing_fixed_by_column: dict[str, int] = {}
    for source in [train_clean_meta, test_clean_meta]:
        for col, count in source["outlier_counts"].items():
            outlier_counts[col] = outlier_counts.get(col, 0) + count
        for col, count in source["missing_fixed_by_column"].items():
            missing_fixed_by_column[col] = missing_fixed_by_column.get(col, 0) + count

    before_df = pd.concat([_with_target(X_train, y_train, target_col), _with_target(X_test, y_test, target_col)])
    cleaned_df = pd.concat(
        [_with_target(cleaned_X_train, y_train, target_col), _with_target(cleaned_X_test, y_test, target_col)]
    ).drop_duplicates()
    cleaning_report = build_cleaning_report(
        before_df,
        cleaned_df,
        {
            "duplicates_before": base_meta["duplicates_before"],
            "standardized_values": base_meta["standardized_values"],
            "normalized_values": base_meta["normalized_values"],
            "type_converted_columns": base_meta["type_converted_columns"],
            "outlier_counts": outlier_counts,
        },
    )
    decisions = {
        "numerical_columns": artifacts["num_cols"],
        "categorical_columns": artifacts["cat_cols"],
        "scaled_columns": artifacts["num_cols"],
        "onehot_columns": artifacts["cat_cols"],
        "missing_fixed_by_column": missing_fixed_by_column,
        "outliers_treated_by_column": outlier_counts,
        "outlier_bounds": artifacts["outlier_bounds"],
        "drop_columns": base_meta["drop_columns"],
    }
    return {
        "cleaned_df": cleaned_df,
        "cleaning_report": cleaning_report,
        "results": {"baseline": baseline_metrics, "cleaned": cleaned_metrics},
        "decisions": decisions,
    }
