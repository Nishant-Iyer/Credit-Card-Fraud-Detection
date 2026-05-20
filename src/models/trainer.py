import logging
import os
import joblib
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import pandas as pd
from typing import Tuple

from src.config import AppConfig
from src.features.transformers import CyclicalTimeEncoder, DataFrameRobustScaler, ColumnDropper
from src.models.autoencoder import AutoencoderFeatureExtractor
from src.models.ensemble import FraudStackingClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def build_training_pipeline(config: AppConfig, use_smote: bool = True) -> Pipeline:
    """
    Builds the end-to-end training pipeline.
    
    0. Column Dropper (removes dropped columns like Time if present)
    1. Cyclical Time Encoder
    2. Robust Scaler (Amount)
    3. PyTorch Autoencoder Feature Extractor (reconstruction_error feature)
    4. SMOTE Resampler (only runs during fit, not predict)
    5. Stacking GBDT Classifier
    """
    logger.info("Building end-to-end ML pipeline...")
    
    # 0. Drop step
    col_dropper = ColumnDropper(drop_cols=config.features.drop_cols)
    
    # 1. Preprocessing steps
    time_encoder = CyclicalTimeEncoder(
        time_col="Time", 
        period=config.features.cyclical_time_period, 
        drop_original=True
    )
    
    scaler = DataFrameRobustScaler(scale_cols=config.features.scale_cols)
    
    # 2. PyTorch Autoencoder representation learner
    ae_extractor = AutoencoderFeatureExtractor(
        input_dim=config.models.autoencoder.input_dim,
        encoding_dim=config.models.autoencoder.encoding_dim,
        epochs=config.models.autoencoder.epochs,
        batch_size=config.models.autoencoder.batch_size,
        learning_rate=config.models.autoencoder.learning_rate,
        model_path=config.models.autoencoder.model_path,
        device="cpu" # Force CPU for local pipeline stability, can change to mps/cuda
    )
    
    # 3. SMOTE (only active during fitting of the pipeline)
    smote_step = SMOTE(random_state=config.data.random_state) if use_smote else None
    
    # 4. Ensemble Classifier
    classifier = FraudStackingClassifier(
        xgb_params={
            "n_estimators": config.models.xgboost.n_estimators,
            "max_depth": config.models.xgboost.max_depth,
            "learning_rate": config.models.xgboost.learning_rate,
            "subsample": config.models.xgboost.subsample,
            "colsample_bytree": config.models.xgboost.colsample_bytree
        },
        lgb_params={
            "n_estimators": config.models.lightgbm.n_estimators,
            "max_depth": config.models.lightgbm.max_depth,
            "num_leaves": config.models.lightgbm.num_leaves,
            "learning_rate": config.models.lightgbm.learning_rate,
            "subsample": config.models.lightgbm.subsample,
            "colsample_bytree": config.models.lightgbm.colsample_bytree
        }
    )
    
    # Assemble pipeline
    steps = [
        ("col_dropper", col_dropper),
        ("time_encoder", time_encoder),
        ("scaler", scaler),
        ("autoencoder", ae_extractor)
    ]
    
    if use_smote:
        steps.append(("smote", smote_step))
        
    steps.append(("classifier", classifier))
    
    pipeline = Pipeline(steps)
    logger.info("Pipeline successfully built.")
    return pipeline


def train_pipeline(
    pipeline: Pipeline, 
    X_train: pd.DataFrame, 
    y_train: pd.Series, 
    save_path: str = "artifacts/full_pipeline.joblib"
) -> Pipeline:
    """
    Fits the pipeline on training data and serializes the result.
    """
    logger.info(f"Starting pipeline fit on training data (shape: {X_train.shape})...")
    pipeline.fit(X_train, y_train)
    logger.info("Pipeline fit complete.")
    
    # Save the pipeline
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(pipeline, save_path)
    logger.info(f"Serialized full pipeline saved to {save_path}")
    
    return pipeline
