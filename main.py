"""
main.py Full diabetes risk prediction pipeline.

You can choose which model to run by setting MODEL_TYPE below:
    MODEL_TYPE = "fnn"       # Feedforward neural net 
    MODEL_TYPE = "knn"       # K-Nearest Neighbours    
    MODEL_TYPE = "softmax"   # Softmax regression      

Usage:
    python main.py
"""
from __future__ import annotations

import numpy as np

from data import SEED, prepare_data
from evaluate import summarize_and_save
from models import fnn_pipeline, knn_pipeline, softmax_pipeline

np.random.seed(SEED)


# Change this string to switch between models.
MODEL_TYPE = "softmax"  # "fnn" | "knn" | "softmax"


def main() -> None:
    # ── 1. Data ───────────────────────────────────────────────────────────────
    print("Loading and preparing data …")
    splits = prepare_data(use_feature_selection=True)
    print(f"Train : {splits.X_train.shape[0]:>7}  features={splits.X_train.shape[1]}")
    print(f"Val   : {splits.X_val.shape[0]:>7}")
    print(f"Test  : {splits.X_test.shape[0]:>7}")
    print(f"Selected features ({len(splits.feature_names)}): {splits.feature_names}\n")

    if MODEL_TYPE == "fnn":
        out = fnn_pipeline(splits)

    elif MODEL_TYPE == "knn":
        out = knn_pipeline(splits)

    elif MODEL_TYPE == "softmax":
        out = softmax_pipeline(splits)

    else:
        raise ValueError(f"Unknown MODEL_TYPE: {MODEL_TYPE}")

    # ── Generic evaluation for any model family ───────────────────────────────
    preds = {
        out["label_baseline"]: out["y_pred_baseline"],
        out["label_balanced"]: out["y_pred_balanced"],
    }
    summarize_and_save(out["y_test"], preds, out["name"])


if __name__ == "__main__":
    main()
