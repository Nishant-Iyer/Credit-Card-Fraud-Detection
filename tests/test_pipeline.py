import numpy as np
import pandas as pd
import pytest
from imblearn.pipeline import Pipeline
from src.config import AppConfig
from src.models.trainer import build_training_pipeline

def test_config_loading(tmp_path):
    # Create dummy yaml config
    yaml_content = """
data:
  openml_id: 1597
  data_dir: "data"
  raw_file: "data/creditcard.csv"
  train_file: "data/train.csv"
  test_file: "data/test.csv"
  target_col: "Class"
  test_size: 0.2
  random_state: 42
features:
  cyclical_time_period: 86400
  scale_cols:
    - "Amount"
  drop_cols:
    - "Time"
models:
  autoencoder:
    input_dim: 3
    encoding_dim: 2
    epochs: 1
    batch_size: 2
    learning_rate: 0.01
    model_path: "tmp_ae.pt"
  xgboost:
    n_estimators: 2
    max_depth: 2
    learning_rate: 0.1
  lightgbm:
    n_estimators: 2
    max_depth: 2
    num_leaves: 3
    learning_rate: 0.1
business_costs:
  manual_review_cost: 5.0
  fraud_cost_factor: 1.0
mlflow:
  tracking_uri: "http://localhost:5000"
  experiment_name: "test"
  run_name: "test_run"
"""
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text(yaml_content)
    
    config = AppConfig.load_from_yaml(str(config_file))
    assert config.data.openml_id == 1597
    assert config.models.autoencoder.input_dim == 3


def test_pipeline_build_and_fit(tmp_path):
    # Define simple training settings
    yaml_content = f"""
data:
  openml_id: 1597
  data_dir: "data"
  raw_file: "data/creditcard.csv"
  train_file: "data/train.csv"
  test_file: "data/test.csv"
  target_col: "Class"
  test_size: 0.2
  random_state: 42
features:
  cyclical_time_period: 86400
  scale_cols:
    - "Amount"
  drop_cols:
    - "Time"
models:
  autoencoder:
    input_dim: 4 # Time_sin, Time_cos, Amount, V1
    encoding_dim: 2
    epochs: 2
    batch_size: 4
    learning_rate: 0.01
    model_path: "{str(tmp_path / 'ae.pt')}"
  xgboost:
    n_estimators: 5
    max_depth: 2
    learning_rate: 0.1
  lightgbm:
    n_estimators: 5
    max_depth: 2
    num_leaves: 4
    learning_rate: 0.1
business_costs:
  manual_review_cost: 5.0
  fraud_cost_factor: 1.0
mlflow:
  tracking_uri: "http://localhost:5000"
  experiment_name: "test"
  run_name: "test_run"
"""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml_content)
    config = AppConfig.load_from_yaml(str(config_file))
    
    pipeline = build_training_pipeline(config, use_smote=True)
    assert isinstance(pipeline, Pipeline)
    
    # Create minimal dummy dataset
    # Need at least 4 rows to support SMOTE fitting (k_neighbors defaults to 5, so we should disable SMOTE or provide enough class size)
    # Actually, we can use use_smote=False for testing fitting to make it easier, or provide 8 rows (6 class 0, 6 class 1)
    df = pd.DataFrame({
        "Time": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0],
        "Amount": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        "V1": [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.9, -1.0, 1.1, -1.2],
        "Class": [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    })
    
    X = df.drop(columns=["Class"])
    y = df["Class"]
    
    pipeline.fit(X, y)
    probs = pipeline.predict_proba(X)[:, 1]
    
    assert len(probs) == len(X)
    assert np.all(probs >= 0.0) & np.all(probs <= 1.0)
