"""Schemas for health and system information endpoints."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health check response schema."""

    status: str = Field(default="healthy", description="Application health status")
    version: str = Field(default="0.1.0", description="Application version")
    timestamp: datetime = Field(description="Current server timestamp")
    database: str = Field(description="Database connectivity status: 'connected' | 'disconnected'")


class SystemInfoResponse(BaseModel):
    """Host and system capability response schema."""

    python_version: str = Field(description="Python interpreter version")
    torch_version: str = Field(description="PyTorch version")
    cpu_count: int = Field(description="Host logical CPU core count")
    torch_cpu_threads: int = Field(description="PyTorch CPU execution threads allocated")
    device_default: str = Field(default="cpu", description="Default tensor device")
    memory_available_mb: float | None = Field(default=None, description="Available host RAM in MB")
    registered_models: list[str] = Field(description="Available embedding model identifiers")
