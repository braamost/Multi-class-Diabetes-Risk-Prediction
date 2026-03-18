"""
Evaluation: metrics and confusion matrix plot.
Mirrors the evaluation section of the diabetes assignment notebook.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

CLASS_NAMES = ["No Diabetes", "Prediabetes", "Diabetes"]


# ─────────────────────────────────────────────────────────────────────────────
#  Core metrics
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Return dict with accuracy, f1_micro, f1_macro, f1_weighted."""
    return {
        "accuracy":    accuracy_score(y_true, y_pred),
        "f1_micro":    f1_score(y_true, y_pred, average="micro"),
        "f1_macro":    f1_score(y_true, y_pred, average="macro"),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
    }


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "",
) -> dict:
    """
    Compute and pretty-print all metrics including the full classification report.
    Returns a dict suitable for building a summary DataFrame.
    """
    acc          = accuracy_score(y_true, y_pred)
    f1_micro     = f1_score(y_true, y_pred, average="micro")
    f1_macro     = f1_score(y_true, y_pred, average="macro")
    f1_weighted  = f1_score(y_true, y_pred, average="weighted")

    print(f'\n{" Model: " + model_name + " ":=^55}')
    print(f"  Accuracy          : {acc:.4f}")
    print(f"  F1 (Micro)        : {f1_micro:.4f}")
    print(f"  F1 (Macro)        : {f1_macro:.4f}")
    print(f"  F1 (Weighted)     : {f1_weighted:.4f}")
    print("\n  Classification Report:")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

    return dict(
        Model=model_name,
        Accuracy=acc,
        F1_Micro=f1_micro,
        F1_Macro=f1_macro,
        F1_Weighted=f1_weighted,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Confusion matrix (row-normalised + raw counts overlay)
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Confusion Matrix",
    ax=None,
) -> None:
    """
    Plot a row-normalised confusion matrix with raw counts in grey below each cell.
    Pass `ax` to embed in a larger figure; otherwise a new figure is created.
    """
    cm      = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(6, 6))

    disp = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=CLASS_NAMES)
    disp.plot(ax=ax, colorbar=False, cmap="Blues", values_format=".2f")
    ax.set_title(title, fontweight="bold")

    # Overlay raw counts
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i + 0.3, f"(n={cm[i, j]:,})",
                ha="center", va="center", fontsize=7, color="grey",
            )

    if own_fig:
        plt.tight_layout()
        plt.show()


def summarize_and_save(
    y_true: np.ndarray,
    predictions: dict,
    model_family: str,
) -> None:
    """
    Given a dict {label -> y_pred}, print metrics, build a summary table,
    and save summary + confusion matrices to disk.
    """
    results = []
    for label, y_pred in predictions.items():
        results.append(compute_metrics(y_true, y_pred, label))

    import pandas as pd

    df = pd.DataFrame(results).set_index("Model").round(4)
    print(f"\n===== {model_family} Summary Table =====")
    print(df.to_string())

    csv_name = f"results_summary_{model_family.lower()}.csv"
    df.to_csv(csv_name)
    print(f"Saved: {csv_name}")

    # Confusion matrix grid
    n = len(predictions)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 5))
    if n == 1:
        axes = [axes]
    for ax, (label, y_pred) in zip(axes, predictions.items()):
        plot_confusion_matrix(y_true, y_pred, title=label, ax=ax)
    plt.suptitle(f"{model_family} – Confusion Matrices (Test Set)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    png_name = f"confusion_matrices_{model_family.lower()}.png"
    plt.savefig(png_name, dpi=150)
    plt.show()
    print(f"Saved: {png_name}")
