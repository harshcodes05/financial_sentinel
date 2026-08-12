from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    PROJECT_NAME: str = "Financial Sentinel API"
    VERSION: str = "2.0.0"  # Updated to v2!
    API_V1_STR: str = "/api/v1"
    ENV: str = "development"
    
    # Model Artifacts (Point to v2!)
    MODEL_PATH: Path = BASE_DIR / "models" / "v2" / "xgboost_model.pkl"
    SCALER_PATH: Path = BASE_DIR / "models" / "v2" / "standard_scaler.pkl"
    
    # API Server Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    model_config = SettingsConfigDict(case_sensitive=True)

settings = Settings()
