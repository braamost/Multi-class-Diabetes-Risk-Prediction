"""
Models and train functions for multi-class diabetes risk prediction.

Each train_* returns (baseline_model, balanced_model).
Imbalance handling is applied inside train via data.resample() for the balanced model.

Section 3 implements:
  - DiabetesFNN  : custom feedforward net with BatchNorm + Dropout
  - SoftF1Loss   : differentiable macro-F1 auxiliary loss
  - train_fnn()  : full training loop with TensorBoard logging (Bonus)
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

from data import SEED, resample

# ── Device ───────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
#  KNN
# ─────────────────────────────────────────────────────────────────────────────

def train_knn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    resample_method: str = "smote",
    n_neighbors: int = 5,
    metric: str = "minkowski",
    random_state: int = SEED,
) -> Tuple[KNeighborsClassifier, KNeighborsClassifier]:
    """Train baseline (original data) and balanced (resampled) KNN."""
    baseline = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric)
    baseline.fit(X_train, y_train)

    X_bal, y_bal = resample(X_train, y_train, method=resample_method, random_state=random_state)
    balanced = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric)
    balanced.fit(X_bal, y_bal)

    return baseline, balanced


# ─────────────────────────────────────────────────────────────────────────────
#  Softmax (multinomial logistic) regression
# ─────────────────────────────────────────────────────────────────────────────

def train_softmax_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    resample_method: str = "smote",
    C: float = 1.0,
    max_iter: int = 1000,
    class_weight: Optional[Dict[int, float]] = None,
    random_state: int = SEED,
) -> Tuple[LogisticRegression, LogisticRegression]:
    """Train baseline and balanced Softmax regression."""
    baseline = LogisticRegression(
        C=C, max_iter=max_iter, multi_class="multinomial", solver="lbfgs",
        class_weight=class_weight,
    )
    baseline.fit(X_train, y_train)

    X_bal, y_bal = resample(X_train, y_train, method=resample_method, random_state=random_state)
    balanced = LogisticRegression(
        C=C, max_iter=max_iter, multi_class="multinomial", solver="lbfgs",
        class_weight=class_weight,
    )
    balanced.fit(X_bal, y_bal)

    return baseline, balanced


# ─────────────────────────────────────────────────────────────────────────────
#  Feedforward Neural Network (PyTorch)
# ─────────────────────────────────────────────────────────────────────────────

class DiabetesFNN(nn.Module):
    """
    Custom feedforward net: BatchNorm → stacked Linear/BN/Activation/Dropout blocks.

    Architecture (default):
        Input → BN →
        Linear(1024) → BN → ReLU → Dropout(0.60) →
        Linear(512)  → BN → ReLU → Dropout(0.50) →
        Linear(256)  → BN → ReLU → Dropout(0.40) →
        Linear(128)  → BN → ReLU → Dropout(0.30) →
        Linear(64)   → BN → ReLU → Dropout(0.15) →
        Linear(32)   → BN → Tanh  → Dropout(0.10) →
        Linear(3)   [logits; softmax handled by CrossEntropyLoss + SoftF1Loss]
    """

    def __init__(
        self,
        n_features: int,
        n_classes: int = 3,
        hidden_dims: Tuple[int, ...] = (1024, 512, 256, 128, 64, 32),
        dropouts: Tuple[float, ...] = (0.60, 0.50, 0.40, 0.30, 0.15, 0.10),
    ) -> None:
        super().__init__()

        layers: List[nn.Module] = [nn.BatchNorm1d(n_features)]
        in_dim = n_features
        for h_dim, drop in zip(hidden_dims, dropouts):
            activation = nn.Tanh() if h_dim == hidden_dims[-1] else nn.ReLU(inplace=True)
            layers += [
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                activation,
                nn.Dropout(drop),
            ]
            in_dim = h_dim

        layers.append(nn.Linear(in_dim, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
#  Soft-F1 auxiliary loss
# ─────────────────────────────────────────────────────────────────────────────

class SoftF1Loss(nn.Module):
    """Differentiable macro soft-F1 loss. Used alongside CrossEntropyLoss."""

    def __init__(self, epsilon: float = 1e-7) -> None:
        super().__init__()
        self.epsilon = epsilon

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.shape[1]
        y_true = nn.functional.one_hot(targets, num_classes).float()
        y_pred = torch.softmax(logits, dim=1)

        tp = (y_true * y_pred).sum(dim=0)
        fp = ((1 - y_true) * y_pred).sum(dim=0)
        fn = (y_true * (1 - y_pred)).sum(dim=0)

        soft_f1 = (2 * tp + self.epsilon) / (2 * tp + fp + fn + self.epsilon)
        return 1 - soft_f1.mean()


# ─────────────────────────────────────────────────────────────────────────────
#  Helper: DataLoader builder
# ─────────────────────────────────────────────────────────────────────────────

def _make_loaders(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_v: np.ndarray,
    y_v: np.ndarray,
    batch_size: int = 512,
) -> Tuple[DataLoader, DataLoader]:
    def _to_ds(X: np.ndarray, y: np.ndarray) -> TensorDataset:
        y_vals = y.values if hasattr(y, "values") else y
        return TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y_vals, dtype=torch.long),
        )

    train_loader = DataLoader(_to_ds(X_tr, y_tr), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(_to_ds(X_v, y_v),   batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


# ─────────────────────────────────────────────────────────────────────────────
#  Training loop
# ─────────────────────────────────────────────────────────────────────────────

def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for Xb, yb in loader:
        Xb, yb = Xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(Xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(yb)
        correct    += (logits.argmax(1) == yb).sum().item()
        n          += len(yb)
    return total_loss / n, correct / n


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion,
    device: torch.device,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    all_preds, all_labels = [], []
    for Xb, yb in loader:
        Xb, yb = Xb.to(device), yb.to(device)
        logits = model(Xb)
        loss   = criterion(logits, yb)
        preds  = logits.argmax(1)
        total_loss += loss.item() * len(yb)
        correct    += (preds == yb).sum().item()
        n          += len(yb)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(yb.cpu().numpy())
    return total_loss / n, correct / n, np.array(all_preds), np.array(all_labels)


@torch.no_grad()
def predict_fnn(
    model: nn.Module,
    X_np: np.ndarray,
    device: torch.device = DEVICE,
    batch_size: int = 1024,
) -> np.ndarray:
    """Return class predictions for a numpy feature matrix."""
    model.eval()
    ds     = TensorDataset(torch.tensor(X_np, dtype=torch.float32))
    loader = DataLoader(ds, batch_size=batch_size)
    preds  = []
    for (Xb,) in loader:
        preds.extend(model(Xb.to(device)).argmax(1).cpu().numpy())
    return np.array(preds)


def _full_train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion,
    optimizer: optim.Optimizer,
    scheduler,
    epochs: int,
    device: torch.device,
    writer: Optional[SummaryWriter] = None,
    tag: str = "",
) -> Dict:
    """Training loop with optional TensorBoard logging (Bonus Section 6)."""
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss, best_state = float("inf"), None

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc         = _train_one_epoch(model, train_loader, criterion, optimizer, device)
        vl_loss, vl_acc, _, _  = _evaluate(model, val_loader, criterion, device)
        if scheduler:
            scheduler.step(vl_loss)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)

        # ── TensorBoard logging (Bonus) ───────────────────────────────────
        if writer:
            writer.add_scalars(f"{tag}/Loss",     {"train": tr_loss, "val": vl_loss}, epoch)
            writer.add_scalars(f"{tag}/Accuracy", {"train": tr_acc,  "val": vl_acc},  epoch)

        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:03d}/{epochs}  "
                f"tr_loss={tr_loss:.4f} tr_acc={tr_acc:.4f}  "
                f"vl_loss={vl_loss:.4f} vl_acc={vl_acc:.4f}"
            )

    model.load_state_dict(best_state)
    return history


# ─────────────────────────────────────────────────────────────────────────────
#  Public API  –  train_fnn
# ─────────────────────────────────────────────────────────────────────────────

def train_fnn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    resample_method: str = "smote",
    input_dim: Optional[int] = None,
    hidden_dims: Optional[Tuple[int, ...]] = None,
    dropouts: Optional[Tuple[float, ...]] = None,
    epochs: int = 50,
    batch_size: int = 512,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    random_state: int = SEED,
    tensorboard_logdir: Optional[str] = "runs",
) -> Tuple[DiabetesFNN, DiabetesFNN]:
    """
    Train baseline and balanced FNN.

    Parameters
    ----------
    X_train, y_train : training data (numpy).
    X_val, y_val     : validation data (numpy) for early stopping / monitoring.
    resample_method  : imbalance strategy for the balanced model ("smote"|"undersample"|"none").
    input_dim        : number of input features; inferred from X_train if None.
    hidden_dims      : hidden layer widths (default: (1024, 512, 256, 128, 64, 32)).
    dropouts         : dropout rates per hidden layer.
    epochs           : training epochs.
    batch_size       : mini-batch size.
    lr               : Adam learning rate.
    weight_decay     : L2 regularisation.
    random_state     : RNG seed.
    tensorboard_logdir : directory for TensorBoard event files (None to disable).

    Returns
    -------
    (baseline_model, balanced_model) – both DiabetesFNN instances on CPU.
    """
    torch.manual_seed(random_state)
    torch.cuda.manual_seed_all(random_state)
    torch.backends.cudnn.deterministic = True

    if input_dim is None:
        input_dim = X_train.shape[1]

    kwargs: Dict = {}
    if hidden_dims is not None:
        kwargs["hidden_dims"] = hidden_dims
    if dropouts is not None:
        kwargs["dropouts"] = dropouts

    # ── Shared val loader ─────────────────────────────────────────────────────
    val_ds     = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val.values if hasattr(y_val, "values") else y_val, dtype=torch.long),
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # ──────────────────────────────────────────────────────────────────────────
    #  Baseline model  (imbalanced, class-weighted CE + soft-F1)
    # ──────────────────────────────────────────────────────────────────────────
    print("=== Training Baseline FNN (Imbalanced Data) ===")

    cw = compute_class_weight(
        "balanced",
        classes=np.array([0, 1, 2]),
        y=y_train.values if hasattr(y_train, "values") else y_train,
    )
    cw_tensor = torch.tensor(cw, dtype=torch.float32).to(DEVICE)

    ce_loss_imb   = nn.CrossEntropyLoss(weight=cw_tensor)
    soft_f1_loss  = SoftF1Loss()

    def combined_loss_imb(logits, targets):
        return ce_loss_imb(logits, targets) + 0.5 * soft_f1_loss(logits, targets)

    train_loader_imb, _ = _make_loaders(X_train, y_train, X_val, y_val, batch_size)

    model_baseline  = DiabetesFNN(input_dim, **kwargs).to(DEVICE)
    optimizer_imb   = optim.Adam(model_baseline.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler_imb   = optim.lr_scheduler.ReduceLROnPlateau(optimizer_imb, patience=5, factor=0.5)

    writer_baseline = (
        SummaryWriter(log_dir=f"{tensorboard_logdir}/FNN_Baseline")
        if tensorboard_logdir else None
    )

    _full_train(
        model_baseline, train_loader_imb, val_loader,
        combined_loss_imb, optimizer_imb, scheduler_imb,
        epochs, DEVICE,
        writer=writer_baseline, tag="Baseline",
    )
    if writer_baseline:
        writer_baseline.close()

    print("\nBaseline FNN training complete.\n")

    # ──────────────────────────────────────────────────────────────────────────
    #  Balanced model  (SMOTE-resampled, plain CE + soft-F1)
    # ──────────────────────────────────────────────────────────────────────────
    print("=== Training Balanced FNN (SMOTE Data) ===")

    X_bal, y_bal = resample(X_train, y_train, method=resample_method, random_state=random_state)
    train_loader_bal, _ = _make_loaders(X_bal, y_bal, X_val, y_val, batch_size)

    ce_loss_bal = nn.CrossEntropyLoss()  # SMOTE already balanced

    def combined_loss_bal(logits, targets):
        return ce_loss_bal(logits, targets) + 0.5 * soft_f1_loss(logits, targets)

    model_balanced  = DiabetesFNN(input_dim, **kwargs).to(DEVICE)
    optimizer_bal   = optim.Adam(model_balanced.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler_bal   = optim.lr_scheduler.ReduceLROnPlateau(optimizer_bal, patience=5, factor=0.5)

    writer_balanced = (
        SummaryWriter(log_dir=f"{tensorboard_logdir}/FNN_SMOTE")
        if tensorboard_logdir else None
    )

    _full_train(
        model_balanced, train_loader_bal, val_loader,
        combined_loss_bal, optimizer_bal, scheduler_bal,
        epochs, DEVICE,
        writer=writer_balanced, tag="SMOTE",
    )
    if writer_balanced:
        writer_balanced.close()

    print("\nBalanced FNN training complete.")

    # Move models to CPU before returning
    return model_baseline.cpu(), model_balanced.cpu()


# ─────────────────────────────────────────────────────────────────────────────
#  High-level pipelines for main.py
# ─────────────────────────────────────────────────────────────────────────────


def fnn_pipeline(splits) -> Dict:
    """
    Run the full FNN pipeline on provided splits and return predictions.

    Does NOT print or plot; main/evaluate handle that.
    """
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
        tensorboard_logdir="runs",
    )

    preds_baseline = predict_fnn(fnn_baseline, splits.X_test)
    preds_balanced = predict_fnn(fnn_balanced, splits.X_test)

    return {
        "name": "FNN",
        "imb_method": "SMOTE",
        "y_test": splits.y_test,
        "y_pred_baseline": preds_baseline,
        "y_pred_balanced": preds_balanced,
        "label_baseline": "FNN Baseline (Imbalanced)",
        "label_balanced": "FNN Balanced (SMOTE)",
    }


def knn_pipeline(splits) -> Dict:
    """
    Full KNN pipeline:
    - Hyperparameter search over k and metric on validation set (imbalanced data).
    - Search best imbalance method (SMOTE vs undersample) on validation.
    - Final predictions for baseline and balanced models on the test set.
    """
    X_train, y_train = splits.X_train, splits.y_train
    X_val, y_val = splits.X_val, splits.y_val
    X_test, y_test = splits.X_test, splits.y_test

    ks = [3, 5, 11, 21]
    metrics = ["euclidean", "manhattan"]
    methods = ["none", "smote"]


    baseline_logs: List[Dict] = []
    imbalance_logs: List[Dict] = []

    best_baseline_score = -1.0
    best_baseline_model: Optional[KNeighborsClassifier] = None

    best_bal_score = -1.0
    best_imb_method = None
    best_balanced_model: Optional[KNeighborsClassifier] = None

    for method in methods:
        if method == "none":
            X_tr, y_tr = X_train, y_train
        else:
            X_tr, y_tr = resample(X_train, y_train, method=method, random_state=SEED)

        for k in ks:
            for metric in metrics:
                model = KNeighborsClassifier(n_neighbors=k, metric=metric)
                model.fit(X_tr, y_tr)
                val_pred = model.predict(X_val)
                f1_macro = float(f1_score(y_val, val_pred, average="macro"))

                row = {
                    "method": method,
                    "k": k,
                    "metric": metric,
                    "val_f1_macro": f1_macro,
                }

                if method == "none":
                    baseline_logs.append(row)
                    if f1_macro > best_baseline_score:
                        best_baseline_score = f1_macro
                        best_baseline_model = model
                else:
                    imbalance_logs.append(row)
                    if f1_macro > best_bal_score:
                        best_bal_score = f1_macro
                        best_imb_method = method
                        best_balanced_model = model

    # Persist tuning logs to inspect why the choices were made.
    if baseline_logs:
        pd.DataFrame(baseline_logs).to_csv("knn_baseline_tuning.csv", index=False)
    if imbalance_logs:
        pd.DataFrame(imbalance_logs).to_csv("knn_imbalance_tuning.csv", index=False)

    preds_baseline = best_baseline_model.predict(X_test)
    preds_balanced = best_balanced_model.predict(X_test)

    return {
        "name": "KNN",
        "imb_method": best_imb_method.upper(),
        "y_test": y_test,
        "y_pred_baseline": preds_baseline,
        "y_pred_balanced": preds_balanced,
        "label_baseline": "KNN Baseline (Imbalanced)",
        "label_balanced": f"KNN Balanced ({best_imb_method.upper()})",
    }

def softmax_pipeline(splits) -> Dict:
    """
    Full Softmax Regression pipeline:
    - Tune C on validation set (by macro-F1)
    - Train baseline (class_weight='balanced') and balanced (SMOTE)
    - Return predictions on test set
    """
    from softmax_regression import tune_softmax_C, train_softmax_regression

    best_C, _ = tune_softmax_C(
        splits.X_train, splits.y_train,
        splits.X_val,   splits.y_val,
    )

    sm_baseline, sm_balanced = train_softmax_regression(
        splits.X_train, splits.y_train,
        C=best_C,
        resample_method="smote",
    )

    return {
        "name": "Softmax",
        "imb_method": "SMOTE",
        "y_test": splits.y_test,
        "y_pred_baseline": sm_baseline.predict(splits.X_test),
        "y_pred_balanced": sm_balanced.predict(splits.X_test),
        "label_baseline": "Softmax Baseline (class_weight='balanced')",
        "label_balanced": "Softmax Balanced (SMOTE)",
    }
