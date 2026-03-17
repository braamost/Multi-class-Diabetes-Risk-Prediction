"""Run data prep, train baseline + balanced models, evaluate."""
from data import SEED, prepare_data
from models import train_knn, train_softmax_regression
from evaluate import evaluate, plot_confusion_matrix
import numpy as np


def main():
    np.random.seed(SEED)

    splits = prepare_data(use_feature_selection=False)
    print("Shapes:", splits.X_train.shape, splits.X_val.shape, splits.X_test.shape)

    # KNN: baseline and balanced (SMOTE)
    knn_baseline, knn_balanced = train_knn(
        splits.X_train, splits.y_train, resample_method="smote", n_neighbors=5
    )
    print("KNN baseline:", evaluate(splits.y_test, knn_baseline.predict(splits.X_test)))
    print("KNN balanced:", evaluate(splits.y_test, knn_balanced.predict(splits.X_test)))
    plot_confusion_matrix(splits.y_test, knn_baseline.predict(splits.X_test), "KNN Baseline")
    plot_confusion_matrix(splits.y_test, knn_balanced.predict(splits.X_test), "KNN Balanced (SMOTE)")

    # Softmax: baseline and balanced
    softmax_baseline, softmax_balanced = train_softmax_regression(
        splits.X_train, splits.y_train, resample_method="smote"
    )
    print("Softmax baseline:", evaluate(splits.y_test, softmax_baseline.predict(splits.X_test)))
    print("Softmax balanced:", evaluate(splits.y_test, softmax_balanced.predict(splits.X_test)))

    # FNN: implement train_fnn() in models.py then uncomment:
    # fnn_baseline, fnn_balanced = train_fnn(splits.X_train, splits.y_train, input_dim=splits.X_train.shape[1])


if __name__ == "__main__":
    main()
