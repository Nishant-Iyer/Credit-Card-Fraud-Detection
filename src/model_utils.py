"""Model utilities for training and evaluation."""
from __future__ import annotations

from typing import Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def train_and_evaluate(
    X, y, test_size: float = 0.2, random_state: int = 42
) -> Tuple[Pipeline, str, float]:
    """Train logistic regression model and compute evaluation metrics.

    Parameters
    ----------
    X, y: array-like
        Features and target data.
    test_size: float, default=0.2
        Proportion of dataset to include in the test split.
    random_state: int, default=42
        Seed used by the random number generator.

    Returns
    -------
    model: Pipeline
        Fitted scikit-learn pipeline.
    report: str
        Textual classification report.
    auc: float
        ROC AUC score on the test split.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(max_iter=1000, class_weight="balanced"),
            ),
        ]
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    y_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)
    return model, report, auc

