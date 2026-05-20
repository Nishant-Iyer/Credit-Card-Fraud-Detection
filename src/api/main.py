from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging
from typing import List, Dict, Any

from src.api.schemas import TransactionInput, TransactionPredictionResponse
from src.pipeline.inference_pipeline import FraudPredictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="Production-grade real-time API for credit card transaction fraud classification.",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model predictor variable
predictor = None

@app.on_event("startup")
def startup_event():
    global predictor
    try:
        # Load predictor
        pipeline_path = os.getenv("MODEL_PATH", "artifacts/full_pipeline.joblib")
        metadata_path = os.getenv("METADATA_PATH", "artifacts/metadata.json")
        predictor = FraudPredictor(pipeline_path=pipeline_path, metadata_path=metadata_path)
        logger.info("FraudPredictor successfully loaded at startup.")
    except Exception as e:
        logger.error(f"Error loading FraudPredictor during startup: {e}")
        # Allow server to start but endpoints will fail gracefully or health check will report unhealthy

@app.get("/health", tags=["Monitoring"])
def health_check() -> Dict[str, Any]:
    """
    Health check endpoint to monitor model load status and service availability.
    """
    global predictor
    status = "healthy"
    model_loaded = predictor is not None and predictor.pipeline is not None
    
    if not model_loaded:
        status = "degraded"
        
    return {
        "status": status,
        "model_loaded": model_loaded,
        "tuned_threshold": predictor.threshold if model_loaded else None
    }

@app.post("/predict", response_model=TransactionPredictionResponse, tags=["Inference"])
def predict_transaction(payload: TransactionInput) -> TransactionPredictionResponse:
    """
    Scores a single transaction for credit card fraud.
    """
    global predictor
    if predictor is None or predictor.pipeline is None:
        raise HTTPException(status_code=503, detail="Model is not loaded or service is degraded.")
        
    try:
        # Convert Pydantic payload to dict
        input_data = payload.dict()
        prediction = predictor.predict_single(input_data)
        
        return TransactionPredictionResponse(
            fraud_probability=prediction["fraud_probability"],
            is_fraud=prediction["is_fraud"],
            threshold_applied=prediction["threshold_applied"],
            status="success"
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/batch", response_model=List[TransactionPredictionResponse], tags=["Inference"])
def predict_batch(payloads: List[TransactionInput]) -> List[TransactionPredictionResponse]:
    """
    Scores a batch of transactions for credit card fraud.
    """
    global predictor
    if predictor is None or predictor.pipeline is None:
        raise HTTPException(status_code=503, detail="Model is not loaded or service is degraded.")
        
    try:
        # Convert list of Pydantic models to pandas DataFrame
        import pandas as pd
        data_dicts = [payload.dict() for payload in payloads]
        df = pd.DataFrame(data_dicts)
        
        preds, probs = predictor.predict(df)
        
        responses = []
        for idx in range(len(payloads)):
            responses.append(
                TransactionPredictionResponse(
                    fraud_probability=float(probs[idx]),
                    is_fraud=bool(preds[idx] == 1),
                    threshold_applied=float(predictor.threshold),
                    status="success"
                )
            )
        return responses
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

# Import os for environment variable inspection
import os

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
