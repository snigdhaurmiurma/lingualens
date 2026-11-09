"""FastAPI application entrypoint for LinguaLens."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.adapters.registry import default_registry
from app.api.v1.api import api_v1_router
from app.db.session import Base, engine

# Setup structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("lingualens.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for database table initialization and graceful cleanup."""
    logger.info("Initializing LinguaLens backend services...")
    # Ensure database schema is initialized
    Base.metadata.create_all(bind=engine)
    logger.info("Registered models: %s", [m.model_id for m in default_registry.list_available_models()])
    yield
    logger.info("Shutting down LinguaLens: releasing model memory...")
    for model_meta in default_registry.list_available_models():
        try:
            adapter = default_registry.get_adapter(model_meta.model_id)
            if adapter.is_loaded:
                adapter.unload()
        except Exception:
            pass
    logger.info("Shutdown complete.")


app = FastAPI(
    title="LinguaLens API",
    description="Multilingual Search Fairness & Evaluation Platform REST & Streaming Gateway",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# CORS configuration for Next.js frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
        "*",  # Local developer mode
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Centralized Exception Handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Sanitized 422 validation error handler."""
    formatted_errors = []
    for err in exc.errors():
        field_path = " -> ".join(str(loc) for loc in err.get("loc", []))
        formatted_errors.append({
            "field": field_path,
            "message": err.get("msg", "Invalid parameter"),
            "type": err.get("type", "validation_error"),
        })

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": True,
            "status_code": 422,
            "message": "Input validation failed",
            "details": formatted_errors,
        },
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Standardized HTTP exception handler."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": True,
            "status_code": exc.status_code,
            "message": exc.detail,
        },
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Sanitized generic 500 handler preventing leakage of server paths or secrets."""
    logger.error("Unhandled server exception: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": True,
            "status_code": 500,
            "message": "Internal server error. Please consult server diagnostics.",
        },
    )


# Mount API v1 router
app.include_router(api_v1_router)


@app.get("/", include_in_schema=False)
def root():
    """Root redirect to OpenAPI documentation."""
    return {
        "service": "LinguaLens API Gateway",
        "version": "0.1.0",
        "documentation": "/docs",
        "health": "/api/v1/health",
    }
