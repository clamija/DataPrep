from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def split_features_target(df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, pd.Series]:
    return df.drop(columns=[target_col]), df[target_col]


def split_train_test(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float,
    random_state: int,
    stratify: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    y_stratify = y if stratify and y.nunique() > 1 else None
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y_stratify)


def infer_feature_types(X_train: pd.DataFrame) -> tuple[list[str], list[str]]:
    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X_train.columns if c not in num_cols]
    return num_cols, cat_cols


def build_preprocessor(num_cols: list[str], cat_cols: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([("scaler", StandardScaler())]), num_cols),
            ("cat", Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
        ]
    )


def build_model_pipeline(preprocessor: ColumnTransformer, random_state: int) -> Pipeline:
    model = LogisticRegression(max_iter=1000, random_state=random_state)
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def fit_and_predict(
    model_pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
) -> pd.Series:
    model_pipeline.fit(X_train, y_train)
    return model_pipeline.predict(X_test)
