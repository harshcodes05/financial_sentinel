from unittest.mock import patch
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_health_endpoint():
    """Test /api/v1/health returns 200 OK."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_readiness_endpoint():
    """Test /api/v1/ready returns ready status when models are loaded."""
    response = client.get("/api/v1/ready")
    assert response.status_code == 200
    assert response.json()["status"] == "ready"

def test_predict_endpoint_success():
    """Test /api/v1/predict returns valid prediction response with dual-model flags."""
    payload = {
        "Time": 0.0,
        "Amount": 149.62,
        "v_features": [0.0] * 28
    }
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["label"] in ["Legitimate", "Fraudulent"]
    assert 0.0 <= data["fraud_probability"] <= 1.0
    assert data["risk_level"] in ["LOW", "MEDIUM", "HIGH"]
    assert "is_anomaly" in data
    assert "consensus_flag" in data

def test_predict_batch_valid_batch_accepts_200():
    """Test /api/v1/predict-batch accepts valid batch array and returns 200 OK."""
    payload = [
        {
            "Time": 0.0,
            "Amount": 149.62,
            "v_features": [0.0] * 28
        },
        {
            "Time": 100.0,
            "Amount": 500.00,
            "v_features": [-2.0] * 28
        }
    ]
    response = client.post("/api/v1/predict-batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    assert "consensus_flag" in data[0]

def test_predict_batch_empty_list_returns_200():
    """Test /api/v1/predict-batch accepts empty list [] and returns [] with 200 OK."""
    payload = []
    response = client.post("/api/v1/predict-batch", json=payload)
    assert response.status_code == 200
    assert response.json() == []

def test_predict_batch_invalid_item_rejected_422():
    """Test /api/v1/predict-batch rejects batch containing an invalid item (e.g. 27 features) with 422."""
    payload = [
        {
            "Time": 0.0,
            "Amount": 100.0,
            "v_features": [0.0] * 28
        },
        {
            "Time": 10.0,
            "Amount": 50.0,
            "v_features": [0.0] * 27
        }
    ]
    response = client.post("/api/v1/predict-batch", json=payload)
    assert response.status_code == 422

def test_api_27_features_rejected_422():
    """Test API rejects transaction payload with 27 v_features with HTTP 422."""
    payload = {
        "Time": 0.0,
        "Amount": 100.0,
        "v_features": [0.0] * 27
    }
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 422

def test_api_29_features_rejected_422():
    """Test API rejects transaction payload with 29 v_features with HTTP 422."""
    payload = {
        "Time": 0.0,
        "Amount": 100.0,
        "v_features": [0.0] * 29
    }
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 422

def test_api_negative_amount_rejected_422():
    """Test API rejects negative transaction amount with HTTP 422."""
    payload = {
        "Time": 0.0,
        "Amount": -50.0,
        "v_features": [0.0] * 28
    }
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 422

def test_api_missing_field_rejected_422():
    """Test API rejects missing required Amount field with HTTP 422."""
    payload = {
        "Time": 0.0,
        "v_features": [0.0] * 28
    }
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 422

def test_api_invalid_type_rejected_422():
    """Test API rejects non-numeric invalid data type with HTTP 422."""
    payload = {
        "Time": 0.0,
        "Amount": "not_a_valid_number",
        "v_features": [0.0] * 28
    }
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 422

def test_api_internal_exception_returns_generic_500():
    """Test API returns generic sanitized error message on internal exception without exposing raw error details."""
    payload = {
        "Time": 0.0,
        "Amount": 100.0,
        "v_features": [0.0] * 28
    }
    with patch("src.api.v1.endpoints.predict.predictor_service.predict", side_effect=RuntimeError("Secret DB Connection Failed")):
        response = client.post("/api/v1/predict", json=payload)
        assert response.status_code == 500
        detail = response.json()["detail"]
        assert detail == "An internal server error occurred during prediction processing."
        assert "Secret DB Connection Failed" not in detail
