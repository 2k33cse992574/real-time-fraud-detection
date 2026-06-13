from fastapi import APIRouter, HTTPException
import joblib
import pandas as pd
import numpy as np
from .schemas import TransactionRequest, PredictionResponse
from ..model.explain import get_explainer, explain_prediction
import os

router = APIRouter()

# Global variables to hold model, preprocessor, explainer
model = None
preprocessor = None
explainer = None
feature_names = []
ordered_columns = [
    'amount', 'transaction_hour', 'merchant_category', 'foreign_transaction',
    'location_mismatch', 'device_trust_score', 'velocity_last_24h', 'cardholder_age'
]

def load_models():
    global model, preprocessor, explainer, feature_names
    model_dir = os.environ.get("MODEL_DIR", ".")
    try:
        model = joblib.load(os.path.join(model_dir, "xgboost_model.joblib"))
        preprocessor = joblib.load(os.path.join(model_dir, "preprocessor.joblib"))
        feature_names = joblib.load(os.path.join(model_dir, "feature_names.joblib"))
        
        # Initialize SHAP explainer (graceful fallback for XGBoost 3.x incompatibility)
        try:
            import shap
            explainer = shap.TreeExplainer(model)
        except Exception as e:
            print(f"SHAP TreeExplainer initialization failed (likely XGBoost 3.x version mismatch): {e}")
            print("Falling back to standard feature importances.")
            explainer = "fallback"
            
        print("Models loaded successfully.")
    except Exception as e:
        print(f"Error loading models: {e}")

@router.post("/predict", response_model=PredictionResponse)
async def predict(transaction: TransactionRequest):
    if model is None or preprocessor is None or explainer is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        # Convert request to pandas DataFrame
        data = pd.DataFrame([transaction.model_dump()])
        
        # Ensure correct column order
        data = data[ordered_columns]
        
        # Scale and Encode the data
        data_transformed = preprocessor.transform(data)
        
        # Predict probability
        probability = float(model.predict_proba(data_transformed)[0][1])
        
        # Boolean prediction (threshold 0.5)
        is_fraud = probability >= 0.5
        
        # SHAP explanation or fallback
        if explainer == "fallback":
            importances = model.feature_importances_.tolist()
            # If a feature importance is 0, we can add tiny jitter so it still renders
            explanation = {"base_value": 0.5, "shap_values": importances, "feature_names": feature_names}
        else:
            explanation = explain_prediction(explainer, data_transformed, feature_names).model_dump()
        
        return PredictionResponse(
            fraud_probability=probability,
            is_fraud=is_fraud,
            explanation=explanation
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
