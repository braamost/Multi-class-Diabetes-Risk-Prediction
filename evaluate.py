"""Evaluation: metrics and confusion matrix plot."""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, f1_score


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Return dict with accuracy, f1_micro, f1_macro, f1_weighted."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_micro": f1_score(y_true, y_pred, average="micro"),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
    }


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, title: str = "Confusion Matrix") -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    plt.show()
