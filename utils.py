from __future__ import annotations

from typing import Iterable

import pandas as pd


def safe_drop_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    cols = [c for c in columns if c in df.columns]
    return df.drop(columns=cols)


def dataframe_profile(df: pd.DataFrame) -> dict:
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "dtypes": df.dtypes.astype(str).to_dict(),
    }

