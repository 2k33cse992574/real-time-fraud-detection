import shap
import joblib
import pandas as pd
import os

MODEL_PATH = "xgboost_model.joblib"

def get_explainer():
    """Loads the XGBoost model and returns a SHAP TreeExplainer."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Train the model first.")
    
    model = joblib.load(MODEL_PATH)
    explainer = shap.TreeExplainer(model)
    return explainer

def explain_prediction(explainer, input_features, feature_names):
    """
    Computes SHAP values for a single prediction.
    input_features: A 2D numpy array or pandas DataFrame (1, num_features)
    """
    # Calculate SHAP values
    shap_values = explainer.shap_values(input_features)
    
    # Base value (expected value)
    base_value = explainer.expected_value
    
    # For a single prediction, shap_values is a 1D array
    if isinstance(shap_values, list): # depending on shap version/model
        shap_values = shap_values[1][0] if len(shap_values) > 1 else shap_values[0]
    elif shap_values.ndim == 2:
        shap_values = shap_values[0]
        
    return {
        "base_value": float(base_value),
        "shap_values": shap_values.tolist(),
        "feature_names": feature_names
    }
