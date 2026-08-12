import pytest
from fastapi.testclient import TestClient
from src.api.main import app
from src.services.predictor import predictor_service

client = TestClient(app)

def test_full_inference_pipeline_trace():
    """Verifies complete inference path: Request -> Pydantic -> Scaler -> XGBoost -> F2 Threshold -> Isolation Forest -> Consensus."""
    payload = {
        "Time": 4462.0,
        "Amount": 239.93,
        "v_features": [
            -2.303350, 1.759247, -0.359745, 2.330243,
            -0.821628, -0.075788, 0.562320, -0.399147,
            -0.238253, -1.525412, 2.032912, -6.560124,
            0.022937, -1.470102, -0.698826, -2.282194,
            -4.781831, -2.615665, -1.334441, -0.430022,
            -0.294166, -0.932391, 0.172726, -0.087330,
            -0.156114, -0.542628, 0.039566, -0.153029
        ]
    }

    # 1. API Call (FastAPI + Pydantic validation)
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 200
    data = response.json()

    # 2. Verify Pydantic Schema Output Keys
    expected_keys = {
        "prediction", "label", "fraud_probability", "confidence",
        "is_anomaly", "anomaly_score", "risk_level", "consensus_flag"
    }
    assert expected_keys.issubset(data.keys())

    # 3. Verify Threshold & Probability Decision Logic
    prob = data["fraud_probability"]
    threshold = predictor_service.threshold
    expected_prediction = 1 if prob >= threshold else 0
    assert data["prediction"] == expected_prediction

    # 4. Verify Label Mapping
    assert data["label"] == ("Fraudulent" if data["prediction"] == 1 else "Legitimate")

    # 5. Verify Risk Level & Consensus Flag Logic
    if data["prediction"] == 1 and data["is_anomaly"]:
        assert data["consensus_flag"] == "CONFIRMED_FRAUD"
        assert data["risk_level"] == "HIGH"
    elif data["prediction"] == 1:
        assert data["consensus_flag"] == "SUPERVISED_FRAUD_FLAG"
    elif data["is_anomaly"]:
        assert data["consensus_flag"] == "ANOMALY_ALERT"
        assert data["risk_level"] == "MEDIUM"
    else:
        assert data["consensus_flag"] == "CLEAN"
        assert data["risk_level"] == "LOW"

def test_streamlit_ui_response_field_alignment():
    """Verifies that all fields rendered by Streamlit UI match the FastAPI PredictionResponse schema."""
    ui_consumed_fields = ["prediction", "label", "fraud_probability", "confidence", "risk_level", "consensus_flag"]
    payload = {
        "Time": 0.0,
        "Amount": 100.0,
        "v_features": [0.0] * 28
    }
    res = client.post("/api/v1/predict", json=payload).json()
    for field in ui_consumed_fields:
        assert field in res, f"UI depends on field '{field}' which is missing in API response!"
