"""Command-line interface for training the fraud detection model."""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib

from data_utils import load_data
from model_utils import train_and_evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a credit card fraud detection model"
    )
    parser.add_argument(
        "--data", type=Path, help="Path to credit card CSV dataset"
    )
    parser.add_argument(
        "--model-out",
        type=Path,
        default=Path("model.joblib"),
        help="Where to save the trained model",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion of dataset used for testing",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    X, y = load_data(args.data)
    model, report, auc = train_and_evaluate(
        X, y, test_size=args.test_size, random_state=args.random_state
    )
    print(report)
    print(f"ROC AUC: {auc:.3f}")
    joblib.dump(model, args.model_out)
    print(f"Model saved to {args.model_out}")


if __name__ == "__main__":
    main()

