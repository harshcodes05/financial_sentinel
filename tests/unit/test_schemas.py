import pytest
from pydantic import ValidationError
from src.schemas.transaction import TransactionInput

def test_valid_transaction_input():
    """Verify valid transaction payload passes schema validation."""
    valid_data = {
        "Time": 100.0,
        "Amount": 50.25,
        "v_features": [0.1] * 28
    }
    tx = TransactionInput(**valid_data)
    assert tx.Time == 100.0
    assert tx.Amount == 50.25
    assert len(tx.v_features) == 28

def test_invalid_amount_raises_error():
    """Verify negative transaction amount is rejected by Pydantic."""
    invalid_data = {
        "Time": 100.0,
        "Amount": -15.0,  # Invalid negative amount
        "v_features": [0.1] * 28
    }
    with pytest.raises(ValidationError):
        TransactionInput(**invalid_data)

def test_invalid_v_features_length_raises_error():
    """Verify v_features list with wrong length (< 28) is rejected."""
    invalid_data = {
        "Time": 100.0,
        "Amount": 50.0,
        "v_features": [0.1] * 10  # Invalid length (10 instead of 28)
    }
    with pytest.raises(ValidationError):
        TransactionInput(**invalid_data)
