import joblib
import json
import os
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Tuple, Union

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class FraudPredictor:
    """
    Inference service class that loads the full trained pipeline
    and applies the optimized decision threshold for predictions.
    """
    def __init__(
        self, 
        pipeline_path: str = "artifacts/full_pipeline.joblib", 
        metadata_path: str = "artifacts/metadata.json"
    ):
        self.pipeline_path = pipeline_path
        self.metadata_path = metadata_path
        self.pipeline = None
        self.threshold = 0.5  # default fallback
        
        self._load_model()

    def _load_model(self):
        # Load Pipeline
        if not os.path.exists(self.pipeline_path):
            raise FileNotFoundError(f"Pipeline file not found at {self.pipeline_path}. Please run training first.")
        
        logger.info(f"Loading pipeline from {self.pipeline_path}...")
        self.pipeline = joblib.load(self.pipeline_path)
        
        # Load Metadata for Threshold
        if os.path.exists(self.metadata_path):
            try:
                with open(self.metadata_path, "r") as f:
                    metadata = json.load(f)
                self.threshold = metadata.get("optimal_threshold", 0.5)
                logger.info(f"Tuned decision threshold loaded: {self.threshold:.4f}")
            except Exception as e:
                logger.warning(f"Error reading metadata from {self.metadata_path}: {e}. Defaulting threshold to 0.5")
        else:
            logger.warning(f"Metadata file not found at {self.metadata_path}. Defaulting threshold to 0.5")

    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predicts fraud binary outcome and probability for a batch of transactions.
        """
        if self.pipeline is None:
            raise RuntimeError("Model pipeline is not loaded.")
            
        # Predict probability (force column ordering)
        cols_order = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
        X_aligned = X[cols_order]
        probs = self.pipeline.predict_proba(X_aligned)[:, 1]
        
        # Classify based on optimal threshold
        preds = (probs >= self.threshold).astype(int)
        
        return preds, probs

    def predict_single(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predicts fraud outcome for a single transaction dictionary.
        """
        # Convert dictionary to DataFrame (ensure columns match pipeline expectations)
        df = pd.DataFrame([transaction])
        
        preds, probs = self.predict(df)
        
        return {
            "fraud_probability": float(probs[0]),
            "is_fraud": bool(preds[0] == 1),
            "threshold_applied": float(self.threshold)
        }
