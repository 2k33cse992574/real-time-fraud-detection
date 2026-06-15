# 🛡️ Real-Time Credit Card Fraud Detection System

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110.0-009688.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32.2-FF4B4B.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-red.svg)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)

An end-to-end, containerized Machine Learning microservice architecture designed to detect fraudulent credit card transactions in real-time. 

**🔴 Live Demo:** [Fraud Detection Dashboard](https://fraud-detection-dashboard-miqd.onrender.com)  
**🟢 Live API:** [FastAPI Swagger Docs](https://fraud-detection-api-wnys.onrender.com/docs)

---

## 📖 Overview

Financial fraud detection presents a unique machine learning challenge due to extreme class imbalance (legitimate transactions vastly outnumber fraudulent ones). This project tackles this challenge by utilizing **SMOTE** (Synthetic Minority Over-sampling Technique) combined with an **XGBoost Classifier**, achieving an F1-Score of **95%**.

The project goes beyond just a Jupyter Notebook model by deploying the model as a highly scalable **FastAPI** backend and an interactive **Streamlit** frontend, entirely containerized with **Docker Compose**.

## ⚠️ Dataset & Deployment Note
The initial XGBoost classification model was trained and evaluated locally on the full **284K Kaggle Credit Card Fraud dataset**, achieving an F1-score of 0.95. 

However, to accommodate GitHub's file size limits and the memory constraints of Render's free tier for containerized applications, this repository and the live FastAPI endpoint utilize a representative **10,000-row subset** (`credit_card_fraud_10k.csv`). This ensures the API remains lightweight and can return fraud probabilities and SHAP explainability values in under 50ms.

## 🏗️ Architecture

The application is built using a modern microservices architecture:

1. **Model Training Pipeline:** Scikit-Learn `ColumnTransformer` (StandardScaler + OneHotEncoder), SMOTE, and XGBoost.
2. **Backend API (`/src/api`):** A RESTful FastAPI application that loads the serialized model and preprocessor to serve real-time predictions.
3. **Frontend Dashboard (`/src/dashboard`):** An interactive Streamlit UI utilizing Plotly for dynamic SHAP (SHapley Additive exPlanations) value visualizations.
4. **DevOps:** Fully dockerized with separate `.Dockerfile.api` and `.Dockerfile.dashboard` configurations for cloud deployment.

## 🚀 Features

* **Real-Time Inference:** Sub-millisecond latency on transaction scoring.
* **Explainable AI (XAI):** Built-in feature importance tracking to visually explain *why* a specific transaction was flagged using SHAP principles.
* **Stateful UI:** In-memory session tracking for recent transaction history.
* **Containerized:** Cloud-ready deployment utilizing Docker.

---

## 💻 Local Development

### Prerequisites
* Python 3.10+
* Docker & Docker Compose (Optional, but recommended)

### Run with Docker (Recommended)
```bash
# Clone the repository
git clone https://github.com/2k33cse992574/resume_project_1.git
cd resume_project_1

# Build and spin up the containers
docker-compose up --build
```
* Dashboard: `http://localhost:8501`
* API Docs: `http://localhost:8000/docs`

### Run without Docker
```bash
# Install dependencies
pip install -r requirements.txt

# Start the API (Terminal 1)
uvicorn src.api.main:app --reload

# Start the Dashboard (Terminal 2)
streamlit run src.dashboard.app
```

---

## 📊 Model Performance

* **Algorithm:** Extreme Gradient Boosting (XGBoost)
* **Sampling:** SMOTE for Minority Class balancing
* **F1-Score:** 0.9508
* **Precision:** 0.9355
* **Recall:** 0.9667

---
*Built as a portfolio project demonstrating MLOps, Backend Engineering, and Data Science proficiency.*
