import pytest
from src.services.predictor import predictor_service
from src.schemas.transaction import TransactionInput

def test_predictor_service_threshold_loaded():
    """Verify optimal threshold is loaded from metadata.json."""
    assert hasattr(predictor_service, "threshold")
    assert isinstance(predictor_service.threshold, float)
    assert 0.0 <= predictor_service.threshold <= 1.0

def test_predictor_service_inference_with_threshold():
    """Verify predictor evaluates prediction using fraud_probability >= threshold."""
    tx = TransactionInput(
        Time=0.0,
        Amount=149.62,
        v_features=[0.0] * 28
    )
    response = predictor_service.predict(tx)
    expected_pred = 1 if response.fraud_probability >= predictor_service.threshold else 0
    assert response.prediction == expected_pred
    assert response.label == ("Fraudulent" if expected_pred == 1 else "Legitimate")
