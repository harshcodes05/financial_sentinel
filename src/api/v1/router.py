from fastapi import APIRouter
from src.api.v1.endpoints import health, predict

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(predict.router)
