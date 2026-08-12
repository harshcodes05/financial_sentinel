import pytest
from pydantic import ValidationError
from src.schemas.transaction import TransactionInput

def test_valid_transaction_input():
    """Verify valid transaction payload with exactly 28 V features passes validation."""
    valid_data = {
        "Time": 100.0,
        "Amount": 50.25,
        "v_features": [0.1] * 28
    }
    tx = TransactionInput(**valid_data)
    assert tx.Time == 100.0
    assert tx.Amount == 50.25
    assert len(tx.v_features) == 28

def test_27_features_rejected():
    """Verify transaction payload with 27 features is rejected."""
    invalid_data = {
        "Time": 100.0,
        "Amount": 50.0,
        "v_features": [0.1] * 27  # 27 features (1 too few)
    }
    with pytest.raises(ValidationError):
        TransactionInput(**invalid_data)

def test_29_features_rejected():
    """Verify transaction payload with 29 features is rejected."""
    invalid_data = {
        "Time": 100.0,
        "Amount": 50.0,
        "v_features": [0.1] * 29  # 29 features (1 too many)
    }
    with pytest.raises(ValidationError):
        TransactionInput(**invalid_data)

def test_negative_amount_rejected():
    """Verify negative transaction amount is rejected."""
    invalid_data = {
        "Time": 100.0,
        "Amount": -15.0,  # Negative amount
        "v_features": [0.1] * 28
    }
    with pytest.raises(ValidationError):
        TransactionInput(**invalid_data)

def test_missing_field_rejected():
    """Verify transaction payload missing required field Amount is rejected."""
    invalid_data = {
        "Time": 100.0,
        # "Amount" is missing
        "v_features": [0.1] * 28
    }
    with pytest.raises(ValidationError):
        TransactionInput(**invalid_data)

def test_invalid_type_rejected():
    """Verify transaction payload with invalid data type for Amount is rejected."""
    invalid_data = {
        "Time": 100.0,
        "Amount": "invalid_string_amount",  # Non-numeric string
        "v_features": [0.1] * 28
    }
    with pytest.raises(ValidationError):
        TransactionInput(**invalid_data)
