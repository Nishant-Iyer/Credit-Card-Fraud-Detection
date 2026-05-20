from pydantic import BaseModel, Field
from typing import Dict, Any

class TransactionInput(BaseModel):
    Time: float = Field(..., description="Seconds elapsed since the first transaction", example=0.0)
    V1: float = Field(..., example=-1.359807)
    V2: float = Field(..., example=-0.072781)
    V3: float = Field(..., example=2.536347)
    V4: float = Field(..., example=1.378155)
    V5: float = Field(..., example=-0.338321)
    V6: float = Field(..., example=0.462388)
    V7: float = Field(..., example=0.239599)
    V8: float = Field(..., example=0.098698)
    V9: float = Field(..., example=0.363787)
    V10: float = Field(..., example=0.090794)
    V11: float = Field(..., example=-0.551600)
    V12: float = Field(..., example=-0.617801)
    V13: float = Field(..., example=-0.991390)
    V14: float = Field(..., example=-0.311169)
    V15: float = Field(..., example=1.468177)
    V16: float = Field(..., example=-0.470401)
    V17: float = Field(..., example=0.207971)
    V18: float = Field(..., example=0.025791)
    V19: float = Field(..., example=0.403993)
    V20: float = Field(..., example=0.251412)
    V21: float = Field(..., example=-0.018307)
    V22: float = Field(..., example=0.277838)
    V23: float = Field(..., example=-0.110474)
    V24: float = Field(..., example=0.066928)
    V25: float = Field(..., example=0.128539)
    V26: float = Field(..., example=-0.189115)
    V27: float = Field(..., example=0.133558)
    V28: float = Field(..., example=-0.021053)
    Amount: float = Field(..., description="Transaction amount in USD", example=149.62)

    class Config:
        json_schema_extra = {
            "example": {
                "Time": 80.0, "V1": -1.359, "V2": -0.072, "V3": 2.536, "V4": 1.378,
                "V5": -0.338, "V6": 0.462, "V7": 0.239, "V8": 0.098, "V9": 0.363,
                "V10": 0.090, "V11": -0.551, "V12": -0.617, "V13": -0.991, "V14": -0.311,
                "V15": 1.468, "V16": -0.470, "V17": 0.207, "V18": 0.025, "V19": 0.403,
                "V20": 0.251, "V21": -0.018, "V22": 0.277, "V23": -0.110, "V24": 0.066,
                "V25": 0.128, "V26": -0.189, "V27": 0.133, "V28": -0.021, "Amount": 149.62
            }
        }


class TransactionPredictionResponse(BaseModel):
    fraud_probability: float = Field(..., description="The probability that the transaction is fraudulent")
    is_fraud: bool = Field(..., description="Binary decision based on optimized cost threshold")
    threshold_applied: float = Field(..., description="The decision threshold used")
    status: str = Field("success", description="Prediction status flag")
