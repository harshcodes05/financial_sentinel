from fastapi import APIRouter, HTTPException, status
from typing import List
from src.schemas.transaction import TransactionInput
from src.schemas.prediction import PredictionResponse
from src.services.predictor import predictor_service
from src.utils.logger import logger

router = APIRouter()

@router.post(
    "/predict", 
    response_model=PredictionResponse, 
    status_code=status.HTTP_200_OK,
    tags=["Inference"]
)
def predict_transaction(transaction: TransactionInput):
    """Predicts risk for a single credit card transaction using the dual-model engine."""
    try:
        return predictor_service.predict(transaction)
    except Exception as e:
        logger.error(f"Prediction API internal error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal server error occurred during prediction processing."
        )

@router.post(
    "/predict-batch", 
    response_model=List[PredictionResponse], 
    status_code=status.HTTP_200_OK,
    tags=["Inference"]
)
def predict_batch(transactions: List[TransactionInput]):
    """Batch prediction endpoint for scoring multiple transactions at once."""
    try:
        results = [predictor_service.predict(tx) for tx in transactions]
        logger.info(f"Processed batch prediction for {len(transactions)} transactions.")
        return results
    except Exception as e:
        logger.error(f"Batch Prediction API internal error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal server error occurred during batch prediction processing."
        )
