"""Schemas for embedding model metadata endpoints."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ModelInfoResponse(BaseModel):
    """Metadata response for an embedding model."""

    model_id: str = Field(description="Unique model identifier")
    display_name: str = Field(description="Human-readable model name")
    provider: str = Field(description="Author or organization")
    parameter_count: int = Field(description="Model parameter count")
    embedding_dimension: int = Field(description="Output vector dimension")
    max_sequence_length: int = Field(description="Maximum context tokens")
    license: str = Field(description="Open-source license")
    query_prefix: str = Field(description="Required prefix for search queries")
    document_prefix: str = Field(description="Required prefix for documents")
    supported_languages: list[str] = Field(description="Supported ISO language codes")
    requires_gpu: bool = Field(default=False, description="Whether GPU is mandatory")
    is_loaded: bool = Field(default=False, description="Whether currently loaded in host memory")
