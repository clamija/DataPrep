from __future__ import annotations

import numpy as np
import pandas as pd


def metrics_comparison_table(results: dict) -> pd.DataFrame:
    rows = []
    for scenario in ["baseline", "cleaned"]:
        m = results[scenario]
        rows.append(
            {
                "Scenario": scenario.title(),
                "Accuracy": round(m["accuracy"], 4),
                "Precision": round(m["precision"], 4),
                "Recall": round(m["recall"], 4),
                "F1": round(m["f1"], 4),
            }
        )
    return pd.DataFrame(rows)


def improvement_text(results: dict) -> str:
    delta = results["cleaned"]["f1"] - results["baseline"]["f1"]
    sign = "+" if delta >= 0 else ""
    n_test = int(results["cleaned"]["n_test"])
    return (
        f"F1 se promijenio za {sign}{delta:.4f} u odnosu na BASELINE "
        f"(odnosno {sign}{delta * 100:.2f} procentnih poena), "
        f"na {n_test} istih test redova."
    )


def cleaning_report_table(cleaning_report: dict) -> pd.DataFrame:
    outliers_treated = int(sum(cleaning_report.get("outlier_counts", {}).values()))
    converted_types = len(cleaning_report.get("type_converted_columns", []))
    return pd.DataFrame(
        [
            {
                "Anomalija u podacima": "Nedostajuće vrijednosti",
                "Stanje prije obrade": cleaning_report["missing_before"],
                "Stanje nakon obrade": cleaning_report["missing_after"],
            },
            {
                "Anomalija u podacima": "Duplikati (redovi)",
                "Stanje prije obrade": cleaning_report.get("duplicates_before", 0),
                "Stanje nakon obrade": 0,
            },
            {
                "Anomalija u podacima": "Neispravni tipovi (broj konvertovanih kolona)",
                "Stanje prije obrade": converted_types,
                "Stanje nakon obrade": 0,
            },
            {
                "Anomalija u podacima": "Ujednačeni tekstualni zapisi (broj izmijenjenih ćelija)",
                "Stanje prije obrade": cleaning_report.get("normalized_values", 0),
                "Stanje nakon obrade": 0,
            },
            {
                "Anomalija u podacima": "Standardizovane varijacije zapisa (broj izmijenjenih ćelija)",
                "Stanje prije obrade": cleaning_report.get("standardized_values", 0),
                "Stanje nakon obrade": 0,
            },
            {
                "Anomalija u podacima": "Ekstremne vrijednosti izvan IQR granica",
                "Stanje prije obrade": outliers_treated,
                "Stanje nakon obrade": 0,
            },
        ]
    )


def decisions_table(decisions: dict) -> pd.DataFrame:
    rows = [
        {
            "Korak pripreme": "Numeričke kolone",
            "Primijenjeno na": ", ".join(decisions["numerical_columns"]) or "-",
        },
        {
            "Korak pripreme": "Kategorijske kolone",
            "Primijenjeno na": ", ".join(decisions["categorical_columns"]) or "-",
        },
        {
            "Korak pripreme": "Skalirane kolone",
            "Primijenjeno na": ", ".join(decisions["scaled_columns"]) or "-",
        },
        {
            "Korak pripreme": "One-hot encoded kolone",
            "Primijenjeno na": ", ".join(decisions["onehot_columns"]) or "-",
        },
        {
            "Korak pripreme": "Dropovane kolone",
            "Primijenjeno na": ", ".join(decisions["drop_columns"]) or "-",
        },
    ]
    return pd.DataFrame(rows)


def per_column_table(values: dict[str, int]) -> pd.DataFrame:
    rows = [{"Kolona": col, "Broj": int(cnt)} for col, cnt in values.items() if int(cnt) > 0]
    if not rows:
        return pd.DataFrame(columns=["Kolona", "Broj"])
    return pd.DataFrame(rows).sort_values("Broj", ascending=False)


def _count_outliers(
    num_df: pd.DataFrame,
    outlier_bounds: dict[str, tuple[float, float]] | None = None,
) -> tuple[int, int]:
    outlier_count = 0
    outlier_cols = 0
    if outlier_bounds is not None:
        columns = [(col, outlier_bounds[col]) for col in outlier_bounds if col in num_df.columns]
    else:
        columns = []
        for col in num_df.columns:
            q1, q3 = num_df[col].quantile(0.25), num_df[col].quantile(0.75)
            iqr = q3 - q1
            if iqr == 0 or pd.isna(iqr):
                continue
            columns.append((col, (q1 - 1.5 * iqr, q3 + 1.5 * iqr)))
    for col, (lower, upper) in columns:
        count = int(((num_df[col] < lower) | (num_df[col] > upper)).sum())
        outlier_count += count
        if count > 0:
            outlier_cols += 1
    return outlier_count, outlier_cols


def data_quality_score(
    df: pd.DataFrame,
    outlier_bounds: dict[str, tuple[float, float]] | None = None,
) -> float:
    rows, cols = df.shape
    total_cells = max(rows * cols, 1)
    missing_cells = int(df.isna().sum().sum())
    missing_ratio = float(missing_cells) / total_cells
    missing_col_ratio = float((df.isna().sum() > 0).sum()) / max(cols, 1)
    duplicate_ratio = float(df.duplicated().sum()) / max(rows, 1)

    invalid_numeric = 0
    invalid_cols = 0
    for col in df.select_dtypes(include=["object"]).columns:
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().mean() >= 0.8:
            invalid_mask = df[col].notna() & converted.isna()
            invalid_count = int(invalid_mask.sum())
            invalid_numeric += invalid_count
            if invalid_count > 0:
                invalid_cols += 1
    invalid_ratio = invalid_numeric / total_cells
    invalid_col_ratio = invalid_cols / max(cols, 1)

    outlier_ratio = 0.0
    outlier_col_ratio = 0.0
    num_df = df.select_dtypes(include=[np.number])
    if not num_df.empty:
        outlier_count, outlier_cols = _count_outliers(num_df, outlier_bounds)
        outlier_ratio = outlier_count / total_cells
        outlier_col_ratio = outlier_cols / max(cols, 1)

    issue_presence_penalty = 0.0
    issue_presence_penalty += 0.04 if missing_cells > 0 else 0.0
    issue_presence_penalty += 0.03 if duplicate_ratio > 0 else 0.0
    issue_presence_penalty += 0.03 if invalid_numeric > 0 else 0.0
    issue_presence_penalty += 0.02 if outlier_ratio > 0 else 0.0

    penalty = (
        (0.24 * missing_ratio)
        + (0.22 * missing_col_ratio)
        + (0.18 * duplicate_ratio)
        + (0.14 * invalid_ratio)
        + (0.10 * invalid_col_ratio)
        + (0.08 * outlier_ratio)
        + (0.04 * outlier_col_ratio)
        + issue_presence_penalty
    )
    return max(0.0, min(100.0, 100.0 * (1.0 - penalty)))
