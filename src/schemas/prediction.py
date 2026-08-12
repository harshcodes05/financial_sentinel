from pydantic import BaseModel, Field

class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="0 for Legitimate, 1 for Fraudulent")
    label: str = Field(..., description="Legitimate or Fraudulent")
    fraud_probability: float = Field(..., description="Probability of transaction being fraudulent (0.0 to 1.0)")
    prediction_confidence: float = Field(..., description="Uncalibrated model certainty score derived from max(p, 1-p)")
    is_anomaly: bool = Field(..., description="True if Isolation Forest detected zero-day anomaly")
    anomaly_score: float = Field(..., description="Unsupervised anomaly decision score")
    risk_level: str = Field(..., description="Risk tier: LOW, MEDIUM, or HIGH")
    consensus_flag: str = Field(..., description="Dual-model agreement: CONFIRMED_FRAUD, ANOMALY_ALERT, CLEAN")
