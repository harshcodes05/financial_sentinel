from pathlib import Path
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    PROJECT_NAME: str = "Financial Sentinel API"
    VERSION: str = "2.0.0"
    API_V1_STR: str = "/api/v1"
    ENV: str = "development"
    
    # Model Artifacts
    MODEL_PATH: Path = BASE_DIR / "models" / "v2" / "xgboost_model.pkl"
    SCALER_PATH: Path = BASE_DIR / "models" / "v2" / "standard_scaler.pkl"
    
    # API Server Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # CORS Origins (Restricted production & development origins)
    CORS_ORIGINS: List[str] = [
        "http://localhost:8501",
        "http://127.0.0.1:8501",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://financialsentinel-5tymssrqdd8rb65bdydg6e.streamlit.app",
        "https://financial-sentinel-16z7.onrender.com",
    ]
    
    model_config = SettingsConfigDict(case_sensitive=True)

settings = Settings()
