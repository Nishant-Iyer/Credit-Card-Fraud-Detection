"""Data loading utilities for credit card fraud detection."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.datasets import make_classification


TARGET_COLUMN = "Class"


def load_data(data_path: Path | None) -> Tuple[pd.DataFrame, pd.Series]:
    """Load the dataset from CSV or generate synthetic data.

    Parameters
    ----------
    data_path: Path | None
        Optional path to a CSV file with the Kaggle credit card dataset. If not
        provided or the file does not exist, a synthetic dataset is generated
        using ``sklearn.datasets.make_classification``.

    Returns
    -------
    X: pd.DataFrame
        Feature matrix.
    y: pd.Series
        Target labels.
    """
    if data_path and data_path.exists():
        data = pd.read_csv(data_path)
        X = data.drop(TARGET_COLUMN, axis=1)
        y = data[TARGET_COLUMN]
    else:
        X_array, y_array = make_classification(
            n_samples=10000,
            n_features=30,
            n_informative=2,
            n_redundant=10,
            n_classes=2,
            weights=[0.99],
            random_state=42,
        )
        columns = [f"feature_{i}" for i in range(X_array.shape[1])]
        X = pd.DataFrame(X_array, columns=columns)
        y = pd.Series(y_array)
    return X, y

