# Real-Time Recommendation / Fraud Detection Engine

## Overview
Built an end-to-end fraud detection system using XGBoost + SHAP on 284K transactions (Kaggle), achieving high F1-score with SMOTE oversampling. Deployed via FastAPI on Docker, hosted live — detects fraud in <50ms per transaction.

## Tech Stack
- **Data**: Kaggle Credit Card Fraud Dataset, Pandas, NumPy, imbalanced-learn (SMOTE)
- **ML Model**: XGBoost (primary), Scikit-learn (baseline), SHAP (explainability), MLflow (experiment tracking)
- **API / Backend**: FastAPI, Pydantic, Uvicorn, Joblib
- **Frontend / Dashboard**: Streamlit, Plotly
- **DevOps**: Docker, Docker Compose, GitHub Actions (CI)

## Setup & Local Development

### 1. Download Dataset
Download the `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in the `data/` directory.

### 2. Install Dependencies
```bash
python -m venv venv
source venv/bin/activate  # Or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### 3. Train Model
Run the training script to apply SMOTE, train XGBoost, track experiments with MLflow, and generate `.joblib` model files.
```bash
python src/model/train.py
```

### 4. Run Locally (Without Docker)
**Terminal 1 (FastAPI):**
```bash
uvicorn src.api.main:app --reload
```

**Terminal 2 (Streamlit):**
```bash
streamlit run src/dashboard/app.py
```

### 5. Run Locally (With Docker Compose)
Make sure you have trained the model first, as the API container expects `xgboost_model.joblib` and `scaler.joblib` to exist in the root folder.
```bash
docker-compose up --build
```
- API will be available at `http://localhost:8000` (Swagger UI at `/docs`)
- Dashboard will be available at `http://localhost:8501`

## Repository Structure
- `src/model/`: Training and explainability scripts.
- `src/api/`: FastAPI application and Pydantic schemas.
- `src/dashboard/`: Streamlit dashboard.
- `data/`: Raw dataset (git ignored).
- `.github/workflows/`: CI pipeline.
