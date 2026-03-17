"""
Models and train functions. Each train_* returns (baseline_model, balanced_model).
Imbalance handling is applied inside train via data.resample() for the balanced model.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import torch
from torch import nn

from data import resample


# --- KNN ---

def train_knn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    resample_method: str = "smote",
    n_neighbors: int = 5,
    metric: str = "minkowski",
    random_state: int = 42,
) -> Tuple[KNeighborsClassifier, KNeighborsClassifier]:
    """Train baseline (original data) and balanced (resampled) KNN."""
    baseline = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric)
    baseline.fit(X_train, y_train)

    X_bal, y_bal = resample(X_train, y_train, method=resample_method, random_state=random_state)
    balanced = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric)
    balanced.fit(X_bal, y_bal)

    return baseline, balanced


# --- Softmax (multinomial logistic) ---

def train_softmax_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    resample_method: str = "smote",
    C: float = 1.0,
    max_iter: int = 1000,
    class_weight: Optional[Dict[int, float]] = None,
    random_state: int = 42,
) -> Tuple[LogisticRegression, LogisticRegression]:
    """Train baseline and balanced Softmax regression."""
    baseline = LogisticRegression(
        C=C, max_iter=max_iter, multi_class="multinomial", solver="lbfgs", class_weight=class_weight
    )
    baseline.fit(X_train, y_train)

    X_bal, y_bal = resample(X_train, y_train, method=resample_method, random_state=random_state)
    balanced = LogisticRegression(
        C=C, max_iter=max_iter, multi_class="multinomial", solver="lbfgs", class_weight=class_weight
    )
    balanced.fit(X_bal, y_bal)

    return baseline, balanced


# --- Feedforward NN (PyTorch skeleton) ---

class FeedForwardNN(nn.Module):
    """Custom feedforward net: configurable hidden dims, ReLU, 3-class softmax output."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Optional[list[int]] = None,
        num_classes: int = 3,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 32]
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_fnn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    resample_method: str = "smote",
    input_dim: Optional[int] = None,
    hidden_dims: Optional[list[int]] = None,
    random_state: int = 42,
) -> Tuple[FeedForwardNN, FeedForwardNN]:
    """
    Skeleton: train baseline and balanced FNN. Implement training loop (optimizer, epochs, loss).
    Optional: TensorBoard logging in the loop.
    """
    # TODO: implement training loop; return (baseline_model, balanced_model)
    raise NotImplementedError("Implement FNN training loop (optimizer, epochs, cross-entropy loss).")
