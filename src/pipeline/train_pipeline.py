import os
import json
import logging
import mlflow
import mlflow.sklearn
import pandas as pd
from typing import Optional

from src.config import AppConfig
from src.data.downloader import download_credit_card_data
from src.data.data_loader import load_and_split_data
from src.models.trainer import build_training_pipeline, train_pipeline
from src.models.evaluator import optimize_business_threshold, evaluate_model_performance, plot_cost_curve

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def run_train_pipeline(config_path: str = "config/config.yaml") -> str:
    """
    Runs the complete training and tuning pipeline end-to-end.
    """
    logger.info("Initializing train pipeline...")
    config = AppConfig.load_from_yaml(config_path)
    
    # 1. Download and split data
    download_credit_card_data(config.data.openml_id, config.data.raw_file)
    train_df, test_df = load_and_split_data(
        raw_path=config.data.raw_file,
        train_path=config.data.train_file,
        test_path=config.data.test_file,
        test_size=config.data.test_size,
        random_state=config.data.random_state,
        target_col=config.data.target_col
    )
    
    X_train = train_df.drop(columns=[config.data.target_col])
    y_train = train_df[config.data.target_col]
    
    X_test = test_df.drop(columns=[config.data.target_col])
    y_test = test_df[config.data.target_col]
    
    # Crucial: Keep raw transaction amounts for business cost valuation
    amounts_test = X_test["Amount"].values
    
    # 2. Setup MLflow Tracking
    mlflow_enabled = False
    try:
        mlflow.set_tracking_uri(config.mlflow.tracking_uri)
        mlflow.set_experiment(config.mlflow.experiment_name)
        mlflow_enabled = True
        logger.info(f"MLflow tracking initialized at: {config.mlflow.tracking_uri}")
    except Exception as e:
        logger.warning(f"Could not connect to MLflow server: {e}. Running pipeline without MLflow logging.")

    # 3. Train and Track
    run_context = mlflow.start_run(run_name=config.mlflow.run_name) if mlflow_enabled else dummy_context()
    
    with run_context as run:
        if mlflow_enabled and run:
            # Log params
            mlflow.log_params({
                "openml_id": config.data.openml_id,
                "test_size": config.data.test_size,
                "random_state": config.data.random_state,
                "ae_epochs": config.models.autoencoder.epochs,
                "ae_encoding_dim": config.models.autoencoder.encoding_dim,
                "xgb_estimators": config.models.xgboost.n_estimators,
                "lgb_estimators": config.models.lightgbm.n_estimators,
                "manual_review_cost": config.business_costs.manual_review_cost,
                "fraud_cost_factor": config.business_costs.fraud_cost_factor
            })
            
        # Build and train
        pipeline = build_training_pipeline(config, use_smote=True)
        pipeline_save_path = "artifacts/full_pipeline.joblib"
        train_pipeline(pipeline, X_train, y_train, save_path=pipeline_save_path)
        
        # 4. Generate Predictions & Optimize Business Threshold
        logger.info("Generating predictions on test set...")
        y_prob_test = pipeline.predict_proba(X_test)[:, 1]
        
        best_threshold, opt_results = optimize_business_threshold(
            y_true=y_test.values,
            y_prob=y_prob_test,
            amounts=amounts_test,
            manual_review_cost=config.business_costs.manual_review_cost,
            fraud_cost_factor=config.business_costs.fraud_cost_factor
        )
        
        # 5. Evaluate and save plots
        metrics = evaluate_model_performance(
            y_true=y_test,
            y_prob=y_prob_test,
            amounts=y_test, # y_test acts as dummy since we pass threshold metrics
            threshold=best_threshold,
            output_dir="artifacts"
        )
        
        plot_cost_curve(opt_results, best_threshold, output_path="artifacts/cost_curve.png")
        
        # 6. Save metadata
        metadata = {
            "optimal_threshold": best_threshold,
            "metrics": metrics,
            "business_metrics": {
                "min_business_cost": opt_results["min_cost"],
                "do_nothing_cost": opt_results["do_nothing_cost"],
                "review_all_cost": opt_results["review_all_cost"],
                "savings_vs_nothing": opt_results["savings_vs_nothing"]
            }
        }
        
        metadata_path = "artifacts/metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)
        logger.info(f"Metadata saved to {metadata_path}")
        
        # 7. Log to MLflow
        if mlflow_enabled:
            # Log metrics
            mlflow.log_metrics({
                "auprc": metrics["auprc"],
                "roc_auc": metrics["roc_auc"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1_score": metrics["f1_score"],
                "optimal_threshold": best_threshold,
                "business_cost": opt_results["min_cost"],
                "business_savings": opt_results["savings_vs_nothing"]
            })
            
            # Log artifacts
            mlflow.log_artifact(metadata_path)
            mlflow.log_artifact("artifacts/pr_curve.png")
            mlflow.log_artifact("artifacts/confusion_matrix.png")
            mlflow.log_artifact("artifacts/cost_curve.png")
            
            # Log model
            mlflow.sklearn.log_model(pipeline, "pipeline")
            logger.info("MLflow logging complete.")
            
    logger.info("Train pipeline execution finished successfully.")
    return "artifacts/full_pipeline.joblib"


class dummy_context:
    """Dummy context manager to use when MLflow is disabled."""
    def __enter__(self):
        return None
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

if __name__ == "__main__":
    run_train_pipeline()
