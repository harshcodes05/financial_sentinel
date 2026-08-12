# 🛡️ Financial Sentinel: Enterprise Credit Card Fraud Detection Microservice

An end-to-end, production-grade machine learning microservice for detecting credit card fraud in real-time. Built with **FastAPI**, **Pydantic v2**, **XGBoost v2**, **Isolation Forest**, **Streamlit**, **Pytest**, and **Docker**.

---

## 🌟 System Architecture

```text
financial_sentinel/
├── src/
│   ├── api/                    # FastAPI REST API layer
│   │   ├── main.py             # Server entrypoint & CORS middleware
│   │   └── v1/
│   │       ├── router.py       # API v1 router
│   │       └── endpoints/      # /health, /ready, /predict, /predict-batch
│   ├── schemas/                # Pydantic v2 data validation schemas
│   │   ├── transaction.py      # TransactionInput schema & bounds checking
│   │   └── prediction.py       # PredictionResponse dual-model schema
│   ├── services/               # Core machine learning inference layer
│   │   ├── preprocessor.py     # Feature scaling transformer
│   │   └── predictor.py        # Dual-engine risk predictor & threshold logic
│   ├── utils/
│   │   └── logger.py           # Structured logging engine
│   └── config.py               # Pydantic-Settings & CORS configuration
├── apps/
│   └── streamlit_app.py        # Enterprise UI Dashboard (Presets, Live latency, CSV batch)
├── scripts/
│   └── train_v2.py             # XGBoost + Isolation Forest model training script
├── tests/
│   ├── unit/                   # Schema & predictor unit tests
│   └── integration/            # FastAPI REST API & full inference path integration tests
├── models/
│   ├── v1/                     # Legacy Random Forest baseline artifacts
│   └── v2/                     # Active XGBoost v2, Isolation Forest & metadata.json
├── notebooks/
│   └── 01_exploration.ipynb   # Exploratory Data Analysis & baseline model exploration
├── Dockerfile.api              # Microservice Docker image definition
├── docker-compose.yml          # Single-command microservice container orchestration
├── pytest.ini                  # Pytest environment & path configuration
└── requirements.txt            # Python dependencies
```

---

## ⚙️ Deployment Modes

The application supports two primary environment modes configured in `src/config.py` via `ENV`:

* **Development Mode (`ENV=development`):**
  * Auto-reload enabled for live local iteration (`uvicorn --reload`).
  * Interactive OpenAPI Swagger documentation active at `http://127.0.0.1:8000/docs` & `http://127.0.0.1:8000/redoc`.
  * Streamlit local dashboard interface active at `http://localhost:8501`.

* **Production Microservice Mode (`ENV=production` / Docker):**
  * Containerized deployment using `Dockerfile.api` and `docker-compose.yml`.
  * Restricted CORS origin whitelisting (`localhost:8501`, `127.0.0.1:8501`, `localhost:8000`, `127.0.0.1:8000`).
  * Sanitized error handling: Full stack traces logged internally (`exc_info=True`) while returning clean, generic HTTP 500 error responses to clients.
  * Dual-Engine inference executing F2-score threshold logic loaded from `metadata.json`.

---

## 📊 Active Model & Benchmarks

The service utilizes a **Dual-Model Risk Engine**:
1. **Supervised Model:** `XGBoostClassifier` trained with cost weighting (`scale_pos_weight = 577.88`) and F2-score decision threshold optimization (`threshold = 0.8854` stored in `models/v2/metadata.json`).
2. **Unsupervised Model:** `IsolationForest` trained on non-fraudulent baseline transactions to detect zero-day behavioral anomalies.

| Metric | Model v1 (Random Forest Baseline) | Model v2 (Active XGBoost + Isolation Forest) | Delta / Benefit |
| :--- | :---: | :---: | :---: |
| **Model Stack** | Single Random Forest | **XGBoost v2 + Isolation Forest** | Dual-Engine Risk Consensus |
| **Imbalance Strategy** | SMOTE | **Cost Weighting (`scale_pos_weight`)** | Direct Loss Matrix Optimization |
| **Fraud Recall** | 72.45% | **84.00%** | **+11.55% More Fraud Caught** |
| **F1 Score** | 0.7717 | **0.8079** | **+3.62% Precision/Recall Balance** |
| **ROC-AUC** | 0.9669 | **0.9734** | **Superior Class Discrimination** |
| **Decision Threshold** | 0.5000 (Default) | **0.8854 (F2-Optimized)** | Loaded dynamically from `metadata.json` |

---

## 🔌 REST API Endpoints

| Endpoint | Method | Description |
| :--- | :---: | :--- |
| `/api/v1/health` | `GET` | Health check endpoint returning HTTP 200 OK |
| `/api/v1/ready` | `GET` | Readiness check confirming model artifacts are loaded |
| `/api/v1/predict` | `POST` | Single transaction inference returning probability & risk consensus |
| `/api/v1/predict-batch` | `POST` | High-throughput batch inference for transaction arrays |

---

## 🔒 Security & Resilience

* **Restricted CORS Policy:** Restricted origins whitelist (`localhost:8501`, `127.0.0.1:8501`, `localhost:8000`, `127.0.0.1:8000`) and allowed methods (`GET`, `POST`, `OPTIONS`).
* **Sanitized Error Responses:** Internal exception stack traces are logged internally (`exc_info=True`) while returning clean, generic error messages to clients.
* **Input Bound Validation:** Pydantic v2 enforces non-negative constraints (`Time >= 0`, `Amount >= 0`) and exact array lengths (28 PCA features).

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

### 3. Start Backend REST API Server
```bash
python -m src.api.main
# API runs on http://127.0.0.1:8000
# OpenAPI Docs: http://127.0.0.1:8000/docs
```

### 4. Start Enterprise Streamlit Dashboard
```bash
streamlit run apps/streamlit_app.py
# UI opens on http://localhost:8501
```

---

## 🐳 Containerized Docker Deployment

To build and run the backend microservice inside an isolated Docker container:

```bash
docker-compose up --build
```

---

## 🛡️ License & Author
* **Author:** Harsh Sharma
* **License:** MIT
