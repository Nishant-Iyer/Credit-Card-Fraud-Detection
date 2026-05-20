import pytest
from fastapi.testclient import TestClient
import numpy as np

# Mock FraudPredictor before importing app to avoid loading actual model files
class MockPipeline:
    def predict_proba(self, X):
        # Return probability array of size (n_samples, 2)
        n = len(X)
        probs = np.zeros((n, 2))
        probs[:, 0] = 0.95 # legit
        probs[:, 1] = 0.05 # fraud
        return probs

class MockPredictor:
    def __init__(self, pipeline_path=None, metadata_path=None):
        self.threshold = 0.25
        self.pipeline = MockPipeline()

    def predict(self, X):
        n = len(X)
        probs = np.array([0.05] * n)
        preds = np.array([0] * n)
        return preds, probs

    def predict_single(self, transaction):
        return {
            "fraud_probability": 0.05,
            "is_fraud": False,
            "threshold_applied": 0.25
        }

# Inject mock into main before importing TestClient
import src.api.main as api_main
api_main.predictor = MockPredictor()

from src.api.main import app

client = TestClient(app)

def test_api_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["tuned_threshold"] == 0.25


def test_api_predict():
    payload = {
        "Time": 100.0,
        "Amount": 50.0,
        **{f"V{i}": 0.0 for i in range(1, 29)}
    }
    
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "fraud_probability" in data
    assert data["is_fraud"] is False
    assert data["threshold_applied"] == 0.25
    assert data["status"] == "success"


def test_api_predict_batch():
    payloads = [
        {
            "Time": 100.0,
            "Amount": 50.0,
            **{f"V{i}": 0.0 for i in range(1, 29)}
        },
        {
            "Time": 200.0,
            "Amount": 150.0,
            **{f"V{i}": 1.0 for i in range(1, 29)}
        }
    ]
    
    response = client.post("/predict/batch", json=payloads)
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    assert data[0]["fraud_probability"] == 0.05
    assert data[1]["is_fraud"] is False
