from fastapi import APIRouter
from src.config import settings

router = APIRouter()

@router.get("/health", tags=["Health"])
def health_check():
    """Health probe indicating server status."""
    return {"status": "ok", "project": settings.PROJECT_NAME, "version": settings.VERSION}

@router.get("/ready", tags=["Health"])
def readiness_check():
    """Readiness probe indicating model artifacts are loaded."""
    from src.services.predictor import predictor_service
    is_ready = predictor_service.model is not None and predictor_service.preprocessor.scaler is not None
    return {"status": "ready" if is_ready else "not_ready"}
