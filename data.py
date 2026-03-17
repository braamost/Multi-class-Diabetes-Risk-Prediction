"""
Data loading, splitting, preprocessing, and imbalance resampling.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# --- Constants (edit paths/seed/target as needed) ---
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CSV_PATH = DATA_DIR / "diabetes_012_health_indicators_BRFSS2015.csv"
SEED = 42
TARGET_COL = "Diabetes_012"
TRAIN_SIZE, VAL_SIZE, TEST_SIZE = 0.7, 0.1, 0.2


@dataclass
class Splits:
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray


def resample(
    X: np.ndarray,
    y: np.ndarray,
    method: str = "smote",
    random_state: int = SEED,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resample (X, y) to mitigate class imbalance.
    method: "smote" | "undersample" | "none"
    """
    if method == "none":
        return X, y
    if method == "smote":
        from imblearn.over_sampling import SMOTE
        return SMOTE(random_state=random_state).fit_resample(X, y)
    if method == "undersample":
        from imblearn.under_sampling import RandomUnderSampler
        return RandomUnderSampler(random_state=random_state).fit_resample(X, y)
    raise ValueError(f"Unknown method: {method}. Use 'smote', 'undersample', or 'none'.")


class FeatureSelector:
    """Skeleton: plug in SelectKBest, SelectFromModel, or RFE later."""

    def __init__(self) -> None:
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FeatureSelector":
        # TODO: fit internal selector (e.g. SelectKBest)
        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("FeatureSelector not fitted.")
        # TODO: return internal selector.transform(X)
        return X

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.fit(X, y)
        return self.transform(X)


def _build_preprocessor(df: pd.DataFrame) -> Pipeline:
    target = TARGET_COL
    feature_cols = [c for c in df.columns if c != target]
    numeric = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    categorical = [c for c in feature_cols if c not in numeric]
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )
    return Pipeline(steps=[("preprocessor", preprocessor)])


def prepare_data(use_feature_selection: bool = False) -> Splits:
    """
    Load CSV, stratified 70/10/20 split, scale numerics, one-hot categoricals.
    Optional feature selection (skeleton).
    """
    df = pd.read_csv(CSV_PATH)
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target '{TARGET_COL}' not in CSV.")

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].values

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, train_size=TRAIN_SIZE, stratify=y, random_state=SEED
    )
    val_ratio = VAL_SIZE / (VAL_SIZE + TEST_SIZE)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=1 - val_ratio, stratify=y_temp, random_state=SEED
    )

    pipe = _build_preprocessor(df)
    X_train = pipe.fit_transform(X_train)
    X_val = pipe.transform(X_val)
    X_test = pipe.transform(X_test)

    if use_feature_selection:
        fs = FeatureSelector()
        X_train = fs.fit_transform(X_train, y_train)
        X_val = fs.transform(X_val)
        X_test = fs.transform(X_test)

    return Splits(
        X_train=X_train, X_val=X_val, X_test=X_test,
        y_train=y_train, y_val=y_val, y_test=y_test,
    )
