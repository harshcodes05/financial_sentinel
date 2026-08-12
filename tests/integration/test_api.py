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
        "v_features": [
            -1.359807, -0.072781, 2.536347, 1.378155,
            -0.338321, 0.462388, 0.239599, 0.098698,
            0.363787, 0.090794, -0.551600, -0.617801,
            -0.991390, -0.311169, 1.468177, -0.470401,
            0.207971, 0.025791, 0.403993, 0.251412,
            -0.018307, 0.277838, -0.110474, 0.066928,
            0.128539, -0.189115, 0.133558, -0.021053
        ]
    }
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["label"] in ["Legitimate", "Fraudulent"]
    assert 0.0 <= data["fraud_probability"] <= 1.0
    assert data["risk_level"] in ["LOW", "MEDIUM", "HIGH"]
    assert "is_anomaly" in data
    assert "consensus_flag" in data

def test_predict_batch_endpoint_success():
    """Test /api/v1/predict-batch accepts array of transactions."""
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
