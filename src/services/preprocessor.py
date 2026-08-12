import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from src.schemas.transaction import TransactionInput
from src.utils.logger import logger

FEATURE_NAMES = [
    "Time",
    "V1", "V2", "V3", "V4", "V5", "V6", "V7",
    "V8", "V9", "V10", "V11", "V12", "V13",
    "V14", "V15", "V16", "V17", "V18", "V19",
    "V20", "V21", "V22", "V23", "V24", "V25",
    "V26", "V27", "V28",
    "Amount"
]

class PreprocessorService:
    def __init__(self, scaler_path: Path):
        self.scaler_path = scaler_path
        self.scaler = self._load_scaler()

    def _load_scaler(self):
        try:
            logger.info(f"Loading scaler artifact from {self.scaler_path}")
            return joblib.load(self.scaler_path)
        except Exception as e:
            logger.error(f"Failed to load scaler from {self.scaler_path}: {e}")
            raise e

    def transform(self, transaction: TransactionInput) -> pd.DataFrame:
        """Constructs ordered feature vector and applies standard scaling."""
        feature_vector = [transaction.Time] + transaction.v_features + [transaction.Amount]
        df = pd.DataFrame([feature_vector], columns=FEATURE_NAMES)
        scaled_array = self.scaler.transform(df)
        return pd.DataFrame(scaled_array, columns=FEATURE_NAMES)
