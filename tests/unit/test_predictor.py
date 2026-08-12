import pytest
from pathlib import Path
from src.services.predictor import PredictorService, predictor_service
from src.schemas.transaction import TransactionInput
from src.schemas.prediction import PredictionResponse

def test_model_loaded_correctly():
    """Verify XGBoost, scaler, and optional Isolation Forest models are loaded into PredictorService."""
    assert predictor_service.model is not None
    assert predictor_service.preprocessor.scaler is not None
    assert hasattr(predictor_service, "threshold")
    assert isinstance(predictor_service.threshold, float)
    assert 0.0 <= predictor_service.threshold <= 1.0

def test_prediction_response_schema_validation():
    """Verify PredictionResponse output schema types and value ranges."""
    tx = TransactionInput(
        Time=0.0,
        Amount=149.62,
        v_features=[0.0] * 28
    )
    res = predictor_service.predict(tx)
    assert isinstance(res, PredictionResponse)
    assert res.prediction in [0, 1]
    assert res.label in ["Legitimate", "Fraudulent"]
    assert 0.0 <= res.fraud_probability <= 1.0
    assert 0.5 <= res.confidence <= 1.0
    assert isinstance(res.is_anomaly, bool)
    assert isinstance(res.anomaly_score, float)
    assert res.risk_level in ["LOW", "MEDIUM", "HIGH"]
    assert res.consensus_flag in ["CONFIRMED_FRAUD", "SUPERVISED_FRAUD_FLAG", "ANOMALY_ALERT", "CLEAN"]

def test_model_artifact_failure_handling():
    """Verify _load_artifact raises exception cleanly on missing non-optional artifact."""
    service = PredictorService()
    non_existent_path = Path("models/v2/non_existent_model.pkl")
    with pytest.raises(Exception):
        service._load_artifact(non_existent_path, "Non-existent model", optional=False)

def test_model_artifact_optional_failure_handling():
    """Verify _load_artifact returns None gracefully on missing optional artifact."""
    service = PredictorService()
    non_existent_path = Path("models/v2/non_existent_optional_model.pkl")
    result = service._load_artifact(non_existent_path, "Optional model", optional=True)
    assert result is None
