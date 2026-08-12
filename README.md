# Financial Sentinel: Credit Card Fraud Detection System
> A containerized fraud detection service combining supervised classification and unsupervised anomaly detection, exposed through FastAPI with an interactive Streamlit dashboard.

Financial Sentinel combines a supervised **XGBoost classifier** with an unsupervised **Isolation Forest anomaly detector**, exposing the inference pipeline through a **FastAPI REST API** and an interactive **Streamlit dashboard**.

## Key Results

- **84.00% fraud recall**, compared with 72.45% for the Random Forest baseline
- **0.8079 F1 score** and **0.9734 ROC-AUC**
- **32.85 ms p50** single-request inference latency
- **24/24 automated tests passing**
- Dockerized FastAPI inference service with batch prediction support

The project focuses on the complete ML-to-application workflow:
- Data preprocessing & scaling
- Imbalanced classification with cost weighting
- Decision-threshold optimization ($F_2$ score)
- Unsupervised anomaly detection
- Strict input schema validation
- Low-latency REST API & batch inference
- Dockerized container deployment

---

## System Architecture

```text
                         User
                          │
                          ▼
                ┌───────────────────┐
                │   Streamlit UI    │
                │      :8501        │
                └─────────┬─────────┘
                          │ HTTP REST
                          ▼
                ┌───────────────────┐
                │   FastAPI API     │
                │      :8000        │
                │    (Dockerized)   │
                └─────────┬─────────┘
                          │
                 ┌────────┴────────┐
                 │                 │
                 ▼                 ▼
          ┌──────────────┐  ┌───────────────┐
          │   XGBoost    │  │ Isolation     │
          │ Classifier   │  │ Forest        │
          └──────┬───────┘  └───────┬───────┘
                 │                  │
                 └────────┬─────────┘
                          ▼
                Risk Aggregation Engine
                          │
                          ▼
                    JSON Response
```

---

## Technology Stack

| Component | Technology |
| :--- | :--- |
| **API Framework** | FastAPI |
| **ASGI Server** | Uvicorn |
| **Data Validation** | Pydantic v2 |
| **Configuration** | Pydantic Settings |
| **Supervised Model** | XGBoost |
| **Anomaly Detector** | Isolation Forest |
| **Preprocessing** | scikit-learn StandardScaler |
| **Frontend** | Streamlit |
| **Data Processing** | Pandas |
| **Testing** | Pytest + FastAPI TestClient |
| **Model Serialization** | Joblib |
| **Containerization** | Docker + Docker Compose |
| **Language** | Python 3.11 / 3.13 |

---

## Project Structure

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
│   │   └── predictor.py        # Dual-model risk prediction & threshold logic
│   ├── utils/
│   │   └── logger.py           # Structured logging engine
│   └── config.py               # Pydantic-Settings & CORS configuration
├── apps/
│   └── streamlit_app.py        # Streamlit UI Dashboard (Presets, Live latency, CSV batch)
├── scripts/
│   ├── train_v2.py             # XGBoost + Isolation Forest model training script
│   └── benchmark_latency.py    # Empirical p50, p95, p99 latency benchmarking script
├── tests/
│   ├── unit/                   # Schema & predictor unit tests
│   └── integration/            # FastAPI REST API & full inference path integration tests
├── models/
│   ├── v1/                     # Legacy Random Forest baseline artifacts
│   └── v2/                     # Active XGBoost, Isolation Forest & metadata
│       ├── xgboost_model.pkl   # Supervised XGBoost classifier
│       ├── isolation_forest_model.pkl # Unsupervised Isolation Forest anomaly detector
│       ├── standard_scaler.pkl # Fitted StandardScaler feature transformer
│       ├── metadata.json       # F2 threshold & model metadata
│       ├── confusion_matrix.png # Reproducible evaluation confusion matrix heatmap
│       └── precision_recall_curve.png # Reproducible F2 Precision-Recall curve
├── notebooks/
│   └── 01_exploration.ipynb    # Exploratory Data Analysis & baseline model exploration
├── Dockerfile.api              # FastAPI service Docker image definition
├── docker-compose.yml           # Container orchestration configuration
├── pytest.ini                  # Pytest environment & path configuration
└── requirements.txt            # Python dependencies
```

---

## Machine Learning Pipeline

Financial Sentinel uses two complementary models:

1. **Supervised XGBoost Classifier:** Learns from labeled historical transactions to estimate fraud probability $P(\mathrm{Fraud} \mid X)$. Predictions are converted to binary fraud decisions using the optimized threshold $\theta^\*$ stored in `models/v2/metadata.json`.

2. **Unsupervised Isolation Forest:** Trained exclusively on legitimate transactions ($y = 0$) to establish a baseline of normal transaction behavior. It flags structural feature outliers using a $0.5\%$ contamination setting selected during training.

### Handling Class Imbalance

Credit card fraud datasets are heavily skewed. Rather than generating synthetic samples using SMOTE, the active XGBoost model uses cost weighting:

$\mathrm{scale\_pos\_weight} = \frac{N_{\mathrm{legit}}}{N_{\mathrm{fraud}}} \approx 577.88$

This assigns substantially greater training weight to fraudulent examples, increasing their contribution to XGBoost's gradient and helping the classifier focus on the minority class without synthetically altering the training distribution.

## Dataset

Financial Sentinel is trained on the public Credit Card Fraud Detection dataset, containing anonymized PCA-transformed transaction features and a highly imbalanced fraud label.

The dataset contains 284,807 transactions, of which 492 are fraudulent.

> The dataset is used for research and demonstration purposes and does not represent real banking transaction data.

### $F_2$-Based Decision Threshold

The default $0.50$ decision threshold is not necessarily optimal for a highly imbalanced fraud detection problem. Because fraud detection prioritizes recall when missed fraud carries greater operational impact than false alerts, the project optimizes the $F_2$ metric:

$$
F_2 = 5 \cdot \frac{\mathrm{Precision} \times \mathrm{Recall}}{4 \cdot \mathrm{Precision} + \mathrm{Recall}}
$$

Selecting the threshold that maximizes $F_2$ on the validation set yields:

$$
\theta^* = 0.8854
$$

The threshold is stored in `models/v2/metadata.json` and loaded dynamically during inference.

---

## Model Performance Benchmarks

Empirical comparison between the legacy Random Forest baseline and the active XGBoost + Isolation Forest stack:

| Metric | Random Forest Baseline | XGBoost Active Stack | Change / Improvement |
| --- | ---: | ---: | --- |
| **Fraud Recall** | 72.45% | **84.00%** | **+11.55 percentage points** |
| **F1 Score** | 0.7717 | **0.8079** | **+0.0362** |
| **ROC-AUC** | 0.9669 | **0.9734** | **+0.0065** |
| **Decision Threshold** | 0.5000 | **0.8854** | $F_2$ optimized |
| **Imbalance Strategy** | SMOTE | **Cost weighting (`scale_pos_weight`)** | Direct loss weighting |

---

## Dual-Model Risk Aggregation Logic

The inference service combines supervised and unsupervised model outputs:

| XGBoost Decision | Isolation Forest | Risk Level | Risk Flag |
| --- | --- | --- | --- |
| **Fraud ($1$)** | **Anomaly (`True`)** | **HIGH** | `CONFIRMED_FRAUD` |
| **Fraud ($1$)** | Normal (`False`) | **HIGH / MEDIUM** | `SUPERVISED_FRAUD_FLAG` |
| Normal ($0$) | **Anomaly (`True`)** | **MEDIUM** | `ANOMALY_ALERT` |
| Normal ($0$) | Normal (`False`) | **LOW** | `CLEAN` |

---

## Prediction Certainty (`prediction_confidence`)

The API exposes `prediction_confidence`, intentionally documented as **uncalibrated**:

$$
\mathrm{prediction}\_\mathrm{confidence} = \max(p, 1-p)
$$
where $p$ is the raw XGBoost fraud probability. It represents the model's absolute distance from the $50/50$ decision boundary rather than a Platt-scaled or isotonic-calibrated probability of correctness.

---

## Inference Latency

Measured using `python -m scripts.benchmark_latency` across 500 single-request runs and 50 batch-request runs, with 100 transactions per batch:

| Benchmark Metric | Mean (ms) | p50 / Median (ms) | p95 (ms) | p99 (ms) |
| :--- | :---: | :---: | :---: | :---: |
| **Single Request (`POST /predict`)** | **37.19 ms** | **32.85 ms** | **56.58 ms** | **113.58 ms** |
| **Batch Request (`POST /predict-batch`, 100 items)** | **3,220.46 ms** | **3,102.80 ms** | **4,506.56 ms** | **5,076.82 ms** |
| **Batch Latency Per Item** | **32.21 ms** | **31.03 ms** | **45.07 ms** | **50.77 ms** |

---

## REST API Endpoints

Base URL: `http://127.0.0.1:8000/api/v1`

### Endpoints Overview

| Endpoint | Method | Description |
| :--- | :---: | :--- |
| `/api/v1/health` | `GET` | Service health status check |
| `/api/v1/ready` | `GET` | Confirms required model artifacts are loaded |
| `/api/v1/predict` | `POST` | Single transaction inference returning probability & risk consensus |
| `/api/v1/predict-batch` | `POST` | Batch inference for transaction arrays |

### Example Request (`POST /api/v1/predict`)
```json
{
  "Time": 4462.0,
  "Amount": 239.93,
  "v_features": [
    -2.303350, 1.759247, -0.359745, 2.330243, -0.821628, -0.075788, 0.562320, -0.399147,
    -0.238253, -1.525412, 2.032912, -6.560124, 0.022937, -1.470102, -0.698826, -2.282194,
    -4.781831, -2.615665, -1.334441, -0.430022, -0.294166, -0.932391, 0.172726, -0.087330,
    -0.156114, -0.542628, 0.039566, -0.153029
  ]
}
```

### Example Response
```json
{
  "prediction": 1,
  "label": "Fraudulent",
  "fraud_probability": 0.94,
  "prediction_confidence": 0.94,
  "is_anomaly": true,
  "anomaly_score": -0.12,
  "risk_level": "HIGH",
  "consensus_flag": "CONFIRMED_FRAUD"
}
```

---

## Streamlit Dashboard

The Streamlit UI frontend (`apps/streamlit_app.py`) provides two interactive workflows:

1. **Single Transaction Inspector:** Profile presets (Legitimate, Medium Risk, High Risk Fraud), manual slider inputs, risk level badges, uncalibrated certainty metrics, and latency measurement.
2. **Batch CSV Inspector:** CSV upload, batch inference via `/predict-batch`, aggregated fraud risk summaries, and downloadable report tables.

Configure the target API backend via environment variable:
```bash
export FINANCIAL_SENTINEL_API_URL="http://127.0.0.1:8000/api/v1"
streamlit run apps/streamlit_app.py
```

---

## Security & Resilience

- **Strict Schema Bounds:** Pydantic v2 enforces non-negative constraints (`Time >= 0`, `Amount >= 0`) and exact array dimensions (28 PCA features).
- **Restricted CORS Policy:** Whitelists allowed origin domains (`localhost:8501`, `127.0.0.1:8501`, `localhost:8000`, `127.0.0.1:8000`) and allowed HTTP methods (`GET`, `POST`, `OPTIONS`).
- **Sanitized Error Responses:** Logs internal stack traces (`exc_info=True`) while returning clean, generic error messages to API clients.
- **Model Readiness Check:** API readiness probe confirms model artifacts are loaded prior to processing inference traffic.

---

## Testing Suite

Execute the full 24-test automated Pytest suite:
```bash
python -m pytest tests/
```
* **Current Status:** **24 / 24 PASSED** (100% test pass rate covering schemas, predictor thresholds, edge cases, and full integration paths).

---

## Local Development & Docker Deployment

### Local Development Setup
```bash
# 1. Clone Repository & Setup Virtual Environment
git clone https://github.com/harshcodes05/financial_sentinel.git
cd financial_sentinel
python -m venv venv
source venv/Scripts/activate  # On Windows Git Bash

# 2. Install Dependencies & Run Tests
pip install -r requirements.txt
python -m pytest tests/

# 3. Start Backend REST API Server
python -m src.api.main
# API runs on http://127.0.0.1:8000 (OpenAPI Docs: http://127.0.0.1:8000/docs)

# 4. Start Streamlit Dashboard (In second terminal)
streamlit run apps/streamlit_app.py
# UI opens on http://localhost:8501
```

### Containerized Docker Deployment
```bash
# Build and start the API container
docker-compose up --build -d

# Check running status
docker ps
```
The Docker container runs Uvicorn without development auto-reloading (`uvicorn src.api.main:app --host 0.0.0.0 --port 8000`), matching the intended deployment configuration.

---

## Project Limitations & Future Scope

### Current Limitations
- Model probabilities are uncalibrated (raw XGBoost scores).
- Isolation Forest anomalies do not guarantee detection of all novel fraud types.
- No persistent database or online retraining pipeline.
- Single-instance lightweight deployment without horizontal scaling.

### Potential Future Enhancements
- Probability calibration using Platt Scaling or Isotonic Regression.
- Model drift monitoring and automated retraining pipelines.
- API authentication, rate limiting, and Prometheus/Grafana observability.
- Performance optimization for larger batch workloads.

---

## Author
* **Author:** Harsh Sharma
* **GitHub:** [harshcodes05](https://github.com/harshcodes05)