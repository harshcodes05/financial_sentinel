import joblib
from pathlib import Path
from src.config import settings
from src.schemas.transaction import TransactionInput
from src.schemas.prediction import PredictionResponse
from src.services.preprocessor import PreprocessorService
from src.utils.logger import logger

class PredictorService:
    def __init__(self):
        self.model_path = settings.MODEL_PATH
        self.scaler_path = settings.SCALER_PATH
        self.iso_path = settings.MODEL_PATH.parent / "isolation_forest_model.pkl"
        
        self.preprocessor = PreprocessorService(self.scaler_path)
        self.model = self._load_artifact(self.model_path, "XGBoost model")
        self.iso_forest = self._load_artifact(self.iso_path, "Isolation Forest model", optional=True)

    def _load_artifact(self, path: Path, description: str, optional: bool = False):
        try:
            logger.info(f"Loading {description} artifact from {path}")
            return joblib.load(path)
        except Exception as e:
            if optional:
                logger.warning(f"Optional artifact {description} not found at {path}: {e}")
                return None
            logger.error(f"Failed to load {description} from {path}: {e}")
            raise e

    def predict(self, transaction: TransactionInput) -> PredictionResponse:
        """Preprocesses transaction and computes dual-model risk consensus."""
        scaled_df = self.preprocessor.transform(transaction)
        
        # 1. Supervised Model (XGBoost)
        prediction = int(self.model.predict(scaled_df)[0])
        fraud_probability = float(self.model.predict_proba(scaled_df)[0][1])
        confidence = float(max(fraud_probability, 1.0 - fraud_probability))

        label = "Fraudulent" if prediction == 1 else "Legitimate"

        # 2. Unsupervised Model (Isolation Forest Anomaly Detector)
        is_anomaly = False
        anomaly_score = 0.0
        if self.iso_forest is not None:
            iso_pred = int(self.iso_forest.predict(scaled_df)[0])
            anomaly_score = float(self.iso_forest.decision_function(scaled_df)[0])
            is_anomaly = (iso_pred == -1)

        # 3. Risk Consensus Logic
        if prediction == 1 and is_anomaly:
            consensus_flag = "CONFIRMED_FRAUD"
            risk_level = "HIGH"
        elif prediction == 1:
            consensus_flag = "SUPERVISED_FRAUD_FLAG"
            risk_level = "HIGH" if fraud_probability >= 0.70 else "MEDIUM"
        elif is_anomaly:
            consensus_flag = "ANOMALY_ALERT"
            risk_level = "MEDIUM"
        else:
            consensus_flag = "CLEAN"
            risk_level = "LOW"

        logger.info(f"Prediction made: label={label}, prob={fraud_probability:.4f}, anomaly={is_anomaly}, consensus={consensus_flag}")

        return PredictionResponse(
            prediction=prediction,
            label=label,
            fraud_probability=fraud_probability,
            confidence=confidence,
            is_anomaly=is_anomaly,
            anomaly_score=anomaly_score,
            risk_level=risk_level,
            consensus_flag=consensus_flag
        )

predictor_service = PredictorService()
