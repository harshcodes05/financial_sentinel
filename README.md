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
│   ├── train_v2.py             # XGBoost + Isolation Forest model training script
│   └── benchmark_latency.py    # Empirical p50, p95, p99 latency benchmarking script
├── tests/
│   ├── unit/                   # Schema & predictor unit tests
│   └── integration/            # FastAPI REST API & full inference path integration tests
├── models/
│   ├── v1/                     # Legacy Random Forest baseline artifacts
│   └── v2/                     # Active XGBoost v2, Isolation Forest & metadata.json
│       ├── xgboost_model.pkl   # Supervised XGBoost classifier
│       ├── isolation_forest_model.pkl # Unsupervised Isolation Forest anomaly detector
│       ├── standard_scaler.pkl # Fitted StandardScaler feature transformer
│       ├── metadata.json       # F2 threshold & model metadata
│       ├── confusion_matrix.png # Reproducible evaluation confusion matrix heatmap
│       └── precision_recall_curve.png # Reproducible F2 Precision-Recall curve
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

## 🌲 Unsupervised Anomaly Engine (Isolation Forest Rationale)

Alongside supervised XGBoost, the microservice integrates an unsupervised **Isolation Forest** anomaly detector.

### 1. 🎯 Why Isolation Forest Exists
Supervised models (XGBoost) excel at matching historical fraud patterns ($y=1$). However, fraudsters constantly design novel evasion strategies (zero-day fraud attacks, unmapped geo-hop patterns). Isolation Forest serves as an **unsupervised zero-day safety net**, isolating abnormal transaction structures that deviate from regular purchasing behaviors even if supervised models assign a low probability.

### 2. 🎛️ What `contamination` Means (`contamination=0.005`)
The `contamination` parameter defines the expected proportion of outliers (anomalies) in the feature space ($0.5\%$). It sets the internal decision boundary $s_0$ for binary classification (`predict(X)` $\rightarrow$ `+1` for normal baseline, `-1` for anomaly). A conservative $0.5\%$ contamination rate ensures high-confidence anomaly flags without inflating false positive alerts.

### 3. 📉 What `decision_function()` Represents
The `decision_function(X)` computes the average isolation depth $s(x, n)$ of a transaction across all random partition trees:
$$s(x, n) = 2^{-\frac{E(h(x))}{c(n)}}$$
* **Negative Scores ($< 0$):** Sample is isolated near tree roots in very few partition splits $\rightarrow$ **High Anomaly Alert**.
* **Positive Scores ($> 0$):** Sample requires deep tree partitioning to isolate $\rightarrow$ **Standard Baseline Purchase**.

### 4. 🛡️ Why Train Exclusively on Legitimate Transactions (`y_train == 0`)
Training Isolation Forest exclusively on verified legitimate transactions establishes a pure mathematical distribution of normal human financial behavior ($P(X \mid Y=0)$). If trained on noisy mixed data, isolation trees would treat historical fraud as part of the normal feature structure. Training on clean data guarantees that **any structural deviation** from normal purchasing behavior triggers an immediate anomaly flag.

---

## 📊 Model Rationale & Evolution

The machine learning pipeline evolved through three major iterations:

### 1. 📈 Why Logistic Regression (Initial Baseline)
* **Role:** Linear statistical benchmark.
* **Rationale:** Established the bare minimum linear decision boundary and served as a sanity check for feature scaling.
* **Limitation:** Incapable of capturing complex non-linear relationships across anonymized PCA components ($V_1 \dots V_{28}$).

### 2. 🌲 Why Random Forest (v1 Baseline)
* **Role:** Non-linear ensemble tree model using bagging.
* **Rationale:** Captured feature interactions and non-linear boundaries while resisting single-tree overfitting.
* **Limitation:** Computationally heavy with slower tree-traversal latency, requiring artificial oversampling (SMOTE) which can distort true joint probabilities.

### 3. ⚡ Why XGBoost (v2 Production Engine)
* **Role:** Gradient Boosted Decision Tree (GBDT) architecture.
* **Rationale:** Sequential tree boosting optimizes second-order Taylor expansion gradients. Consistently outperforms Random Forest in both execution speed and classification accuracy on tabular financial data.
* **Advantage:** Native cache-aware memory vectorization and direct loss-matrix scaling without needing synthetic data generation.

### 4. ⚖️ Why `scale_pos_weight`
* **Role:** Loss-matrix weighting factor ($\text{scale\_pos\_weight} = \frac{N_{\text{legit}}}{N_{\text{fraud}}} \approx 577.88$).
* **Rationale:** Rather than generating synthetic transactions (SMOTE) which alters empirical data distributions, `scale_pos_weight` penalizes false negatives $577.88\times$ more heavily directly inside XGBoost's log-loss gradient:
  $$L(\theta) = - \sum \left[ w_i \cdot y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]$$

### 5. 🎯 Why $F_2$ Score Optimization
* **Role:** Asymmetric evaluation metric weighting Recall higher than Precision ($\beta = 2.0$):
  $$F_2 = (1 + 2^2) \frac{\text{Precision} \times \text{Recall}}{(2^2 \times \text{Precision}) + \text{Recall}} = 5 \cdot \frac{\text{Precision} \times \text{Recall}}{4 \cdot \text{Precision} + \text{Recall}}$$
* **Rationale:** In financial fraud prevention, an uncaught fraudulent transaction (False Negative) results in direct chargeback losses and regulatory fines ($1,000+), whereas a false alert (False Positive) only costs a minor review fee ($<1). $F_2$ explicitly prioritizes catching fraud over precision.

### 6. 🎛️ Why Threshold Tuning
* **Role:** Decision boundary adjustment ($p \ge \theta^*$) instead of arbitrary default $0.50$.
* **Rationale:** The default $0.50$ threshold is mathematically suboptimal for skewed distributions and asymmetric loss matrices. Scanning the Precision-Recall curve yields $\theta^* = 0.8854$ (saved in `models/v2/metadata.json`), elevating Fraud Recall to **84.00%** (+11.55% improvement over baseline).

---

## 📈 Model Performance Benchmarks

| Metric | Model v1 (Random Forest Baseline) | Model v2 (Active XGBoost + Isolation Forest) | Delta / Benefit |
| :--- | :---: | :---: | :---: |
| **Model Stack** | Single Random Forest | **XGBoost v2 + Isolation Forest** | Dual-Engine Risk Consensus |
| **Imbalance Strategy** | SMOTE | **Cost Weighting (`scale_pos_weight`)** | Direct Loss Matrix Optimization |
| **Fraud Recall** | 72.45% | **84.00%** | **+11.55% More Fraud Caught** |
| **F1 Score** | 0.7717 | **0.8079** | **+3.62% Precision/Recall Balance** |
| **ROC-AUC** | 0.9669 | **0.9734** | **Superior Class Discrimination** |
| **Decision Threshold** | 0.5000 (Default) | **0.8854 (F2-Optimized)** | Loaded dynamically from `metadata.json` |

---

## ⏱️ Empirical Inference Latency Benchmarks

Empirically measured via `python -m scripts.benchmark_latency` across 500 single-request and 50 batch-request (100 items each) HTTP invocations:

| Benchmark Metric | Mean (ms) | p50 / Median (ms) | p95 (ms) | p99 (ms) |
| :--- | :---: | :---: | :---: | :---: |
| **Single Request (`POST /predict`)** | **37.19 ms** | **32.85 ms** | **56.58 ms** | **113.58 ms** |
| **Batch Request (`POST /predict-batch`, 100 items)** | **3220.46 ms** | **3102.80 ms** | **4506.56 ms** | **5076.82 ms** |
| **Batch Latency Per Item** | **32.21 ms** | **31.03 ms** | **45.07 ms** | **50.77 ms** |

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

### 3. Run Latency Benchmark Suite
```bash
python -m scripts.benchmark_latency
```

### 4. Start Backend REST API Server
```bash
python -m src.api.main
# API runs on http://127.0.0.1:8000
# OpenAPI Docs: http://127.0.0.1:8000/docs
```

### 5. Start Enterprise Streamlit Dashboard
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
