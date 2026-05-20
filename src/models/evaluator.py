import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_curve, auc, classification_report, confusion_matrix
import logging
import os
from typing import Dict, Any, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def calculate_business_cost(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    amounts: np.ndarray,
    manual_review_cost: float = 5.0,
    fraud_cost_factor: float = 1.0
) -> float:
    """
    Computes total business cost:
    - Every predicted fraud (True Positive + False Positive) incurs manual review cost.
    - Every undetected fraud (False Negative) incurs the transaction amount * fraud cost factor.
    """
    # True Positives & False Positives (predicted positive)
    predicted_positives = (y_pred == 1)
    review_costs = np.sum(predicted_positives) * manual_review_cost
    
    # False Negatives (actual positive, predicted negative)
    false_negatives = (y_true == 1) & (y_pred == 0)
    fraud_loss = np.sum(amounts[false_negatives] * fraud_cost_factor)
    
    total_cost = review_costs + fraud_loss
    return float(total_cost)


def optimize_business_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    amounts: np.ndarray,
    manual_review_cost: float = 5.0,
    fraud_cost_factor: float = 1.0
) -> Tuple[float, Dict[str, Any]]:
    """
    Finds the probability decision threshold that minimizes the business cost.
    """
    thresholds = np.linspace(0.001, 0.999, 200)
    costs = []
    
    # Cost of doing nothing (predicting everything is legit)
    do_nothing_loss = np.sum(amounts[y_true == 1] * fraud_cost_factor)
    
    # Cost of reviewing everything (blocking/reviewing all)
    review_all_cost = len(y_true) * manual_review_cost
    
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        cost = calculate_business_cost(y_true, y_pred, amounts, manual_review_cost, fraud_cost_factor)
        costs.append(cost)
        
    best_idx = np.argmin(costs)
    best_threshold = thresholds[best_idx]
    min_cost = costs[best_idx]
    
    # Compare savings
    savings_vs_nothing = do_nothing_loss - min_cost
    
    logger.info(f"Optimal Threshold: {best_threshold:.4f} | Min Business Cost: ${min_cost:,.2f}")
    logger.info(f"Do Nothing Cost: ${do_nothing_loss:,.2f} | Savings: ${savings_vs_nothing:,.2f}")
    
    return float(best_threshold), {
        "thresholds": thresholds,
        "costs": costs,
        "min_cost": min_cost,
        "do_nothing_cost": do_nothing_loss,
        "review_all_cost": review_all_cost,
        "savings_vs_nothing": savings_vs_nothing
    }


def evaluate_model_performance(
    y_true: pd.Series,
    y_prob: np.ndarray,
    amounts: pd.Series,
    threshold: float,
    output_dir: str = "artifacts"
) -> Dict[str, Any]:
    """
    Evaluates classification performance, plots curves, and returns metrics dictionary.
    """
    os.makedirs(output_dir, exist_ok=True)
    y_pred = (y_prob >= threshold).astype(int)
    
    # Basic metrics
    auprc = average_precision_score(y_true, y_prob)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    logger.info(f"AUPRC: {auprc:.4f} | ROC-AUC: {roc_auc:.4f} | F1: {f1:.4f}")
    logger.info(f"Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    
    # 1. Plot Precision-Recall Curve
    plt.figure(figsize=(8, 6))
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    plt.plot(rec, prec, label=f"Stacking Ensemble (AUPRC = {auprc:.4f})", color="darkorange", lw=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.savefig(f"{output_dir}/pr_curve.png", dpi=300)
    plt.close()
    
    # 2. Plot Confusion Matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Legit", "Fraud"], yticklabels=["Legit", "Fraud"])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix (Threshold = {threshold:.3f})")
    plt.savefig(f"{output_dir}/confusion_matrix.png", dpi=300)
    plt.close()
    
    return {
        "auprc": float(auprc),
        "roc_auc": float(roc_auc),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp)
        }
    }


def plot_cost_curve(
    opt_results: Dict[str, Any],
    best_threshold: float,
    output_path: str = "artifacts/cost_curve.png"
):
    """
    Plots the Business Cost vs Threshold curve.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(opt_results["thresholds"], opt_results["costs"], label="Business Cost ($)", color="crimson", lw=2)
    plt.axvline(best_threshold, color="green", linestyle="--", label=f"Optimal Threshold ({best_threshold:.3f})")
    plt.axhline(opt_results["do_nothing_cost"], color="gray", linestyle=":", label="Do Nothing Cost")
    
    plt.xlabel("Decision Probability Threshold")
    plt.ylabel("Total Financial Loss ($)")
    plt.title("Business Cost Optimization vs. Decision Threshold")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved business cost curve plot to {output_path}")
