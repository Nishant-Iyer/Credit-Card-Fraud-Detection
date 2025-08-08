"""Evaluate a saved fraud detection model on a dataset."""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
from sklearn.metrics import classification_report, roc_auc_score

from data_utils import load_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained credit card fraud detection model"
    )
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to credit card CSV dataset",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("model.joblib"),
        help="Path to the trained model",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    X, y = load_data(args.data)
    model = joblib.load(args.model)
    y_pred = model.predict(X)
    report = classification_report(y, y_pred)
    y_proba = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, y_proba)
    print(report)
    print(f"ROC AUC: {auc:.3f}")


if __name__ == "__main__":
    main()

