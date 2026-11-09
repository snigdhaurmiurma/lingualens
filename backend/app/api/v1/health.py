"""Health, system capability, and model catalog endpoints."""

from __future__ import annotations

import platform
import sys
from datetime import datetime, timezone

import torch
from fastapi import APIRouter, Depends, status
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.adapters.registry import default_registry
from app.db.session import get_db
from app.schemas.health import HealthResponse, SystemInfoResponse
from app.schemas.model import ModelInfoResponse

router = APIRouter(tags=["Health & System"])


@router.get("/health", response_model=HealthResponse, status_code=status.HTTP_200_OK)
def get_health(db: Session = Depends(get_db)) -> HealthResponse:
    """Return health status and database connectivity."""
    db_status = "connected"
    try:
        db.execute(text("SELECT 1;"))
    except Exception:
        db_status = "disconnected"

    return HealthResponse(
        status="healthy" if db_status == "connected" else "degraded",
        version="0.1.0",
        timestamp=datetime.now(timezone.utc),
        database=db_status,
    )


@router.get("/system/info", response_model=SystemInfoResponse, status_code=status.HTTP_200_OK)
def get_system_info() -> SystemInfoResponse:
    """Return host execution environment details."""
    cpu_count = os_cpu = platform.processor() or "Unknown CPU"
    logical_cores = torch.get_num_threads()

    return SystemInfoResponse(
        python_version=sys.version.split()[0],
        torch_version=torch.__version__,
        cpu_count=logical_cores,
        torch_cpu_threads=logical_cores,
        device_default="cpu",
        memory_available_mb=None,
        registered_models=[m.model_id for m in default_registry.list_available_models()],
    )


@router.get("/models", response_model=list[ModelInfoResponse], status_code=status.HTTP_200_OK)
def list_models() -> list[ModelInfoResponse]:
    """List all registered multilingual embedding models with their metadata."""
    models = default_registry.list_available_models()
    responses: list[ModelInfoResponse] = []

    for m in models:
        adapter = default_registry.get_adapter(m.model_id)
        responses.append(
            ModelInfoResponse(
                model_id=m.model_id,
                display_name=m.display_name,
                provider=m.provider,
                parameter_count=m.parameter_count,
                embedding_dimension=m.embedding_dimension,
                max_sequence_length=m.max_sequence_length,
                license=m.license,
                query_prefix=m.query_prefix,
                document_prefix=m.document_prefix,
                supported_languages=m.supported_languages,
                requires_gpu=m.requires_gpu,
                is_loaded=adapter.is_loaded,
            )
        )

    return responses
