"""
main.py – Full diabetes risk prediction pipeline.

Runs:
  1. Data preparation (with feature selection)
  2. Train FNN: baseline (imbalanced + class weights) and balanced (SMOTE)
  3. Evaluation: accuracy, F1 scores, confusion matrices
  4. Summary table + learning curves
  5. TensorBoard logs (bonus)

Usage:
    python main.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data  import SEED, prepare_data
from models import train_fnn, predict_fnn
from evaluate import compute_metrics, plot_confusion_matrix

np.random.seed(SEED)


def main() -> None:
    # ── 1. Data ───────────────────────────────────────────────────────────────
    print("Loading and preparing data …")
    splits = prepare_data(use_feature_selection=True)
    print(f"Train : {splits.X_train.shape[0]:>7}  features={splits.X_train.shape[1]}")
    print(f"Val   : {splits.X_val.shape[0]:>7}")
    print(f"Test  : {splits.X_test.shape[0]:>7}")
    print(f"Selected features ({len(splits.feature_names)}): {splits.feature_names}\n")

    # ── 2. Train FNN (baseline + balanced) ───────────────────────────────────
    fnn_baseline, fnn_balanced = train_fnn(
        X_train=splits.X_train,
        y_train=splits.y_train,
        X_val=splits.X_val,
        y_val=splits.y_val,
        resample_method="smote",
        input_dim=splits.X_train.shape[1],
        epochs=50,
        batch_size=512,
        lr=1e-3,
        weight_decay=1e-4,
        tensorboard_logdir="runs",  # TensorBoard bonus
    )

    # ── 3. Evaluation ─────────────────────────────────────────────────────────
    preds_baseline = predict_fnn(fnn_baseline, splits.X_test)
    preds_balanced = predict_fnn(fnn_balanced, splits.X_test)

    results = []
    results.append(compute_metrics(splits.y_test, preds_baseline, "FNN Baseline (Imbalanced)"))
    results.append(compute_metrics(splits.y_test, preds_balanced, "FNN Balanced (SMOTE)"))

    # ── 4. Confusion matrices ─────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    plot_confusion_matrix(splits.y_test, preds_baseline, "FNN Baseline (Imbalanced)", axes[0])
    plot_confusion_matrix(splits.y_test, preds_balanced, "FNN Balanced (SMOTE)",      axes[1])
    plt.suptitle("Confusion Matrices — Test Set (Row-Normalized)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("confusion_matrices.png", dpi=150)
    plt.show()
    print("Saved: confusion_matrices.png")

    # ── 5. Summary table ──────────────────────────────────────────────────────
    results_df = pd.DataFrame(results).set_index("Model").round(4)
    print("\n===== Summary Table =====")
    print(results_df.to_string())
    results_df.to_csv("results_summary.csv")
    print("Saved: results_summary.csv")

    print(
        "\nTensorBoard logs are in ./runs/\n"
        "Run:  tensorboard --logdir=runs\n"
        "Then open http://localhost:6006"
    )


if __name__ == "__main__":
    main()
