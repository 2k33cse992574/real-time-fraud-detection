import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import mlflow
import joblib

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "../../"))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "creditcard.csv")
MODEL_DIR = PROJECT_ROOT

# Features Lists
NUMERICAL_FEATURES = [
    'amount', 'transaction_hour', 'foreign_transaction',
    'location_mismatch', 'device_trust_score', 'velocity_last_24h', 'cardholder_age'
]
CATEGORICAL_FEATURES = ['merchant_category']

def load_and_preprocess_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}. Please download it from Kaggle.")
    
    df = pd.read_csv(DATA_PATH)
    
    # Drop irrelevant identifier
    if 'transaction_id' in df.columns:
        X = df.drop(['transaction_id', 'is_fraud'], axis=1)
    else:
        X = df.drop(['is_fraud'], axis=1)
        
    y = df['is_fraud']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Build a ColumnTransformer for preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), NUMERICAL_FEATURES),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL_FEATURES)
        ]
    )
    
    # Fit and transform
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)
    
    # Apply SMOTE to handle class imbalance
    print("Applying SMOTE...")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_transformed, y_train)
    
    # Save the preprocessor for the API
    joblib.dump(preprocessor, os.path.join(MODEL_DIR, "preprocessor.joblib"))
    
    # Extract actual feature names after transformation for SHAP
    # Get categorical feature names
    cat_encoder = preprocessor.named_transformers_['cat']
    cat_feature_names = cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES).tolist()
    
    # Combine numerical and categorical feature names
    all_feature_names = NUMERICAL_FEATURES + cat_feature_names
    joblib.dump(all_feature_names, os.path.join(MODEL_DIR, "feature_names.joblib"))
    
    return X_train_resampled, y_train_resampled, X_test_transformed, y_test

def train_baseline(X_train, y_train, X_test, y_test):
    print("Training Baseline (Logistic Regression)...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    f1 = f1_score(y_test, preds)
    print(f"Baseline F1-Score: {f1:.4f}")
    return model, f1

def train_xgboost(X_train, y_train, X_test, y_test):
    print("Training XGBoost...")
    
    mlflow.set_experiment("Fraud_Detection_Experiment")
    
    with mlflow.start_run():
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': 5,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'random_state': 42
        }
        
        mlflow.log_params(params)
        
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)
        
        preds = model.predict(X_test)
        
        f1 = f1_score(y_test, preds)
        precision = precision_score(y_test, preds, zero_division=0)
        recall = recall_score(y_test, preds, zero_division=0)
        
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        
        print(f"XGBoost F1-Score: {f1:.4f}")
        print(f"XGBoost Precision: {precision:.4f}")
        print(f"XGBoost Recall: {recall:.4f}")
        
        # Save confusion matrix
        cm = confusion_matrix(y_test, preds)
        joblib.dump(cm, os.path.join(MODEL_DIR, "confusion_matrix.joblib"))
        
        # Log model in mlflow
        mlflow.xgboost.log_model(model, "xgboost_model")
        
        # Save model for API
        joblib.dump(model, os.path.join(MODEL_DIR, "xgboost_model.joblib"))
        
        return model

if __name__ == "__main__":
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    X_train, y_train, X_test, y_test = load_and_preprocess_data()
    
    train_baseline(X_train, y_train, X_test, y_test)
    train_xgboost(X_train, y_train, X_test, y_test)
    print("Training complete. Models and preprocessor saved.")
