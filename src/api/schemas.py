from pydantic import BaseModel, Field
from typing import List

class TransactionRequest(BaseModel):
    amount: float = Field(default=0.0)
    transaction_hour: int = Field(default=12, ge=0, le=23)
    merchant_category: str = Field(default="Grocery")
    foreign_transaction: int = Field(default=0, ge=0, le=1)
    location_mismatch: int = Field(default=0, ge=0, le=1)
    device_trust_score: int = Field(default=50, ge=0, le=100)
    velocity_last_24h: int = Field(default=1)
    cardholder_age: int = Field(default=30)

class SHAPExplanation(BaseModel):
    base_value: float
    shap_values: List[float]
    feature_names: List[str]

class PredictionResponse(BaseModel):
    fraud_probability: float
    is_fraud: bool
    explanation: SHAPExplanation
