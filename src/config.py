import os
import yaml
from typing import List, Optional
from pydantic import BaseModel, Field

class DataConfig(BaseModel):
    openml_id: int
    data_dir: str
    raw_file: str
    train_file: str
    test_file: str
    target_col: str
    test_size: float = 0.2
    random_state: int = 42

class FeaturesConfig(BaseModel):
    cyclical_time_period: float = 86400.0
    scale_cols: List[str]
    drop_cols: List[str]

class AutoencoderConfig(BaseModel):
    input_dim: int
    encoding_dim: int = 16
    epochs: int = 15
    batch_size: int = 256
    learning_rate: float = 0.001
    model_path: str = "artifacts/autoencoder.pt"

class XGBoostConfig(BaseModel):
    n_estimators: int = 200
    max_depth: int = 6
    learning_rate: float = 0.1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    scale_pos_weight: float = 1.0

class LightGBMConfig(BaseModel):
    n_estimators: int = 200
    max_depth: int = 6
    num_leaves: int = 31
    learning_rate: float = 0.1
    subsample: float = 0.8
    colsample_bytree: float = 0.8

class ModelsConfig(BaseModel):
    autoencoder: AutoencoderConfig
    xgboost: XGBoostConfig
    lightgbm: LightGBMConfig

class BusinessCostsConfig(BaseModel):
    manual_review_cost: float = 5.0
    fraud_cost_factor: float = 1.0

class MLflowConfig(BaseModel):
    tracking_uri: str
    experiment_name: str
    run_name: str

class AppConfig(BaseModel):
    data: DataConfig
    features: FeaturesConfig
    models: ModelsConfig
    business_costs: BusinessCostsConfig
    mlflow: MLflowConfig

    @classmethod
    def load_from_yaml(cls, yaml_path: str) -> "AppConfig":
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")
        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
