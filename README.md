# 🛡️ Financial Sentinel: Enterprise Credit Card Fraud Detection Microservice

An end-to-end, production-grade machine learning microservice for detecting fraudulent credit card transactions in real-time. Built with **FastAPI**, **Pydantic v2**, **XGBoost**, **Isolation Forest**, **Streamlit**, **Pytest**, and **Docker**.

---

## 🌟 System Architecture

```text
financial_sentinel/
├── config/                     # Environment configuration loader
├── src/
│   ├── config.py               # Pydantic-Settings & model registry pathing
│   ├── utils/logger.py         # Structured logging engine
│   ├── schemas/                # Pydantic data validation layer
│   ├── services/               # Preprocessor & Dual-Model inference engine
│   └── api/                    # FastAPI REST API layer (/health, /ready, /predict, /predict-batch)
├── apps/
│   └── streamlit_app.py        # Enterprise UI Dashboard (Live latency tracking, preset profiles, batch CSV)
├── scripts/
│   └── train_v2.py             # Automated XGBoost + Isolation Forest training script
├── tests/
│   ├── unit/                   # Schema & predictor unit tests
│   └── integration/            # FastAPI HTTP endpoint tests
├── models/
│   ├── v1/                     # Legacy Random Forest artifacts
│   └── v2/                     # Active XGBoost v2 + Isolation Forest & metadata.json
├── Dockerfile.api              # Backend Docker image definition
└── docker-compose.yml          # Container orchestration
```

---

## 📊 Model Performance Benchmarks

| Metric | Model v1 (Random Forest) | Model v2 (XGBoost + Isolation Forest) | Improvement |
| :--- | :---: | :---: | :---: |
| **Model Stack** | Random Forest | **XGBoost + Isolation Forest** | Dual-Engine Risk Consensus |
| **Imbalance Strategy** | SMOTE | **Cost Weighting (`scale_pos_weight`)** | Direct Loss Tuning |
| **Fraud Recall** | 72.45% | **84.00%** | **+11.55% More Fraud Caught** |
| **F1 Score** | 0.7717 | **0.8079** | **+3.62% Precision/Recall Balance** |
| **ROC-AUC** | 0.9669 | **0.9734** | **Higher Discrimination** |
| **Decision Threshold** | 0.5000 | **0.8854 (F2-Optimized)** | Loaded from `metadata.json` |

---

## 🚀 Quick Start Guide

### 1. Environment Setup
```bash
# Clone Repository
git clone https://github.com/harshcodes05/financial_sentinel.git
cd financial_sentinel

# Create & Activate Virtual Environment
python -m venv venv
source venv/Scripts/activate  # On Windows Git Bash

# Install Production Dependencies
pip install -r requirements.txt
```

### 2. Run Automated Test Suite
```bash
python -m pytest tests/
```

### 3. Start Backend API Server
```bash
python -m src.api.main
# Server runs on http://127.0.0.1:8000
# OpenAPI Docs: http://127.0.0.1:8000/docs
```

### 4. Start Enterprise Streamlit Dashboard
```bash
streamlit run apps/streamlit_app.py
# UI opens on http://localhost:8501
```

---

## 🐳 Docker Deployment

To build and run the backend microservice inside an isolated Linux container:

```bash
docker-compose up --build
```

---

## 🛡️ License & Author
* **Author:** Harsh Sharma
* **License:** MIT
