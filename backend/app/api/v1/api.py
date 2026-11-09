"""API v1 root router aggregating sub-routers."""

from fastapi import APIRouter

from app.api.v1.datasets import router as datasets_router
from app.api.v1.experiments import router as experiments_router
from app.api.v1.health import router as health_router
from app.api.v1.search import router as search_router
from app.api.v1.tokenization import router as tokenization_router

api_v1_router = APIRouter(prefix="/api/v1")

api_v1_router.include_router(health_router)
api_v1_router.include_router(datasets_router)
api_v1_router.include_router(tokenization_router)
api_v1_router.include_router(search_router)
api_v1_router.include_router(experiments_router)
