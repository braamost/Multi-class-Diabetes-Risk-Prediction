from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from data import prepare_data, resample, SEED
from evaluate import compute_metrics, plot_confusion_matrix


def predict_softmax(model: LogisticRegression, X: np.ndarray) -> np.ndarray:
    return model.predict(X)


# Hyperparameter tuning: find best C (regularization parameter) = 1/λ
def tune_softmax_C(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    C_values: list[float] | None = None,
    max_iter: int = 200,
    random_state: int = SEED,
) -> tuple[float, pd.DataFrame]:
    """
    Selects the C that maximises macro-F1 on the validation set
    (macro-F1 is the most informative metric for imbalanced multi-class data).

    Returns
    -------
    best_C : float
    results_df : DataFrame with columns [C, val_accuracy, val_f1_macro]
    """
    if C_values is None:
        C_values = [0.01, 0.1, 1.0, 10.0]

    rows = []
    print("─" * 55)
    print(f"{'C':>10}  {'Val Accuracy':>14}  {'Val F1-Macro':>13}")
    print("─" * 55)

    for C in C_values:
        model = LogisticRegression(
            C=C,
            max_iter=max_iter,
            multi_class="multinomial",
            solver="lbfgs",
            n_jobs=-1,
            random_state=random_state,
        )
        model.fit(X_train, y_train)
        y_pred_val = model.predict(X_val)

        acc = accuracy_score(y_val, y_pred_val)
        f1_macro = f1_score(y_val, y_pred_val, average="macro")

        rows.append({"C": C, "val_accuracy": acc, "val_f1_macro": f1_macro})
        print(f"  C={C:>8}  acc={acc:.4f}        macro-F1={f1_macro:.4f}")

    print("─" * 55)

    results_df = pd.DataFrame(rows)
    best_C = float(results_df.loc[results_df["val_f1_macro"].idxmax(), "C"])
    print(f"\n  → Best C = {best_C}  (highest val macro-F1)\n")
    return best_C, results_df


def train_softmax_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    resample_method: str = "smote",
    C: float = 1.0,
    max_iter: int = 200,
    random_state: int = SEED,
) -> tuple[LogisticRegression, LogisticRegression]:
    """
    Returns
    -------
    (baseline_model, balanced_model)
    """

    # Baseline: imbalanced data + class_weight='balanced'
    print("=== Training Baseline Softmax (Imbalanced + class_weight='balanced') ===")
    baseline = LogisticRegression(
        C=C,
        max_iter=max_iter,
        multi_class="multinomial",
        class_weight="balanced",  # compensates for imbalance via loss weighting
        solver="lbfgs",
        n_jobs=-1,
        random_state=random_state,
    )
    baseline.fit(X_train, y_train)
    print("  Baseline training complete.\n")

    # Balanced: SMOTE-resampled data
    print(f"=== Training Balanced Softmax ({resample_method.upper()} resampled) ===")
    X_bal, y_bal = resample(
        X_train, y_train, method=resample_method, random_state=random_state
    )
    print(
        f"  Resampled training set size: {len(y_bal):,}  "
        f"(classes: {dict(zip(*np.unique(y_bal, return_counts=True)))})"
    )

    balanced = LogisticRegression(
        C=C,
        max_iter=max_iter,
        multi_class="multinomial",
        solver="lbfgs",
        n_jobs=-1,
        random_state=random_state,
    )
    balanced.fit(X_bal, y_bal)
    print("  Balanced training complete.\n")

    return baseline, balanced


def plot_tuning_curve(results_df: pd.DataFrame, best_C: float) -> None:
    """Plot validation macro-F1 and accuracy vs. log(C)."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogx(
        results_df["C"],
        results_df["val_f1_macro"],
        marker="o",
        label="Val Macro-F1",
        color="steelblue",
    )
    ax.semilogx(
        results_df["C"],
        results_df["val_accuracy"],
        marker="s",
        linestyle="--",
        label="Val Accuracy",
        color="darkorange",
    )
    ax.axvline(
        best_C, color="red", linestyle=":", linewidth=1.5, label=f"Best C = {best_C}"
    )
    ax.set_xlabel("C  (log scale)", fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Softmax Regression – Hyperparameter Tuning (C)", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("softmax_tuning_curve.png", dpi=150)
    plt.show()
    print("Saved: softmax_tuning_curve.png")


def run_softmax_section(
    splits,
    resample_method: str = "smote",
    C_values: list[float] | None = None,
    max_iter: int = 200,
) -> list[dict]:
    """
    Returns
    -------
    List of metric dicts (baseline + balanced) for the summary table.
    """
    print("[ Step 1 ] Hyperparameter Tuning — searching best C …\n")
    best_C, tuning_df = tune_softmax_C(
        splits.X_train,
        splits.y_train,
        splits.X_val,
        splits.y_val,
        C_values=C_values,
        max_iter=max_iter,
    )
    plot_tuning_curve(tuning_df, best_C)

    print("[ Step 2 ] Training with best C …\n")
    sm_baseline, sm_balanced = train_softmax_regression(
        splits.X_train,
        splits.y_train,
        resample_method=resample_method,
        C=best_C,
        max_iter=max_iter,
    )

    preds_baseline = predict_softmax(sm_baseline, splits.X_test)
    preds_balanced = predict_softmax(sm_balanced, splits.X_test)

    print("[ Step 3 ] Test-set Evaluation\n")
    results = []
    results.append(
        compute_metrics(
            splits.y_test, preds_baseline, "Softmax Baseline (class_weight='balanced')"
        )
    )
    results.append(
        compute_metrics(
            splits.y_test,
            preds_balanced,
            f"Softmax Balanced ({resample_method.upper()})",
        )
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    plot_confusion_matrix(
        splits.y_test,
        preds_baseline,
        "Softmax Baseline (class_weight='balanced')",
        axes[0],
    )
    plot_confusion_matrix(
        splits.y_test,
        preds_balanced,
        f"Softmax Balanced ({resample_method.upper()})",
        axes[1],
    )
    plt.suptitle(
        "Softmax Regression — Confusion Matrices (Row-Normalized)",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig("softmax_confusion_matrices.png", dpi=150)
    plt.show()
    print("Saved: softmax_confusion_matrices.png")

    return results


if __name__ == "__main__":
    print("Loading and preparing data …")
    # Let's turn off polynomial features to avoid the noise and keep it fast
    splits = prepare_data(use_feature_selection=True, use_polynomial_features=True)
    print(f"Train : {splits.X_train.shape[0]:>7}   features={splits.X_train.shape[1]}")
    print(f"Val   : {splits.X_val.shape[0]:>7}")
    print(f"Test  : {splits.X_test.shape[0]:>7}\n")

    # Switch to undersampling instead of SMOTE
    results = run_softmax_section(splits, resample_method="undersample")

    # Summary table
    summary_df = pd.DataFrame(results).set_index("Model").round(4)
    print("\n===== Softmax Regression — Summary Table =====")
    print(summary_df.to_string())
    summary_df.to_csv("softmax_results_summary.csv")
    print("Saved: softmax_results_summary.csv")
