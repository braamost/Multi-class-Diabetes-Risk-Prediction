"""
Data loading, splitting, preprocessing, and imbalance resampling.
Implements the diabetes_012_health_indicators_BRFSS2015 dataset pipeline.
"""

from __future__ import annotations

import random
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

warnings.filterwarnings("ignore")

# --- Constants ---
ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "diabetes_012_health_indicators_BRFSS2015.csv"
SEED = 42
TARGET_COL = "Diabetes_012"
K_FEATURES = 15  # Top-K features via mutual information

# Fixed seeds for reproducibility
random.seed(SEED)
np.random.seed(SEED)


@dataclass
class Splits:
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    feature_names: List[str] = None  # selected feature names (if FS used)


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
    if method == "smotetomek":
        from imblearn.combine import SMOTETomek

        return SMOTETomek(random_state=random_state).fit_resample(X, y)
    if method == "smoteenn":
        from imblearn.combine import SMOTEENN

        return SMOTEENN(random_state=random_state).fit_resample(X, y)
    raise ValueError(
        f"Unknown method: {method}. Use 'smote', 'undersample', 'smotetomek', 'smoteenn', or 'none'."
    )


class FeatureSelector:
    """
    SelectKBest with mutual_info_classif — selects top-K most informative features.
    Fit only on training set to avoid data leakage.
    """

    def __init__(self, k: int = K_FEATURES) -> None:
        self.k = k
        self._selector = SelectKBest(score_func=mutual_info_classif, k=k)
        self._fitted = False
        self.selected_mask_: Optional[np.ndarray] = None
        self.scores_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FeatureSelector":
        self._selector.fit(X, y)
        self.selected_mask_ = self._selector.get_support()
        self.scores_ = self._selector.scores_
        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("FeatureSelector not fitted.")
        return self._selector.transform(X)

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.fit(X, y)
        return self.transform(X)


def prepare_data(
    use_feature_selection: bool = True, use_polynomial_features: bool = False
) -> Splits:
    """
    Load CSV → stratified 70/10/20 split → StandardScaler → optional SelectKBest → optional PolynomialFeatures.

    Returns a Splits dataclass with numpy arrays ready for model training.
    """
    # ── 1. Load ──────────────────────────────────────────────────────────────
    df = pd.read_csv(CSV_PATH)
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target '{TARGET_COL}' not in CSV.")

    TARGET = TARGET_COL
    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(int)
    feature_names_all = X.columns.tolist()

    # ── 2. Split (70 train / 10 val / 20 test, stratified) ──────────────────
    # First split off 20 % test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=SEED
    )
    # Then split remaining 80 % → 70 % train + 10 % val  (val = 10/80 = 0.125)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.125, stratify=y_trainval, random_state=SEED
    )

    # ── 3. Standard Scaling (fit only on train) ───────────────────────────────
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # ── 4. Feature Selection (optional) ──────────────────────────────────────
    selected_names = feature_names_all  # default: all
    if use_feature_selection:
        fs = FeatureSelector(k=K_FEATURES)
        X_train_scaled = fs.fit_transform(X_train_scaled, y_train.values)
        X_val_scaled = fs.transform(X_val_scaled)
        X_test_scaled = fs.transform(X_test_scaled)
        selected_names = [
            feature_names_all[i] for i, keep in enumerate(fs.selected_mask_) if keep
        ]

    # ── 5. Polynomial Features (optional) ────────────────────────────────────
    if use_polynomial_features:
        poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
        X_train_scaled = poly.fit_transform(X_train_scaled)
        X_val_scaled = poly.transform(X_val_scaled)
        X_test_scaled = poly.transform(X_test_scaled)
        # We can update selected_names if needed, but we'll just append dummy names for now
        selected_names = poly.get_feature_names_out(selected_names)

    return Splits(
        X_train=X_train_scaled,
        X_val=X_val_scaled,
        X_test=X_test_scaled,
        y_train=y_train.values,
        y_val=y_val.values,
        y_test=y_test.values,
        feature_names=selected_names,
    )
