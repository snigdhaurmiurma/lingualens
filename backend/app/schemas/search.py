"""Schemas for semantic vector search endpoints."""

from __future__ import annotations

import uuid
from typing import Any
from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    """Payload to execute semantic search across indexed corpus documents."""

    query_text: str = Field(min_length=1, max_length=1000, description="Query text to search for")
    model_id: str = Field(default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", description="Embedding model ID or alias ('minilm', 'e5-small')")
    retrieval_mode: str = Field(default="EXACT", pattern=r"^(EXACT|VECTOR_INDEX)$", description="Search mode: 'EXACT' (brute-force) or 'VECTOR_INDEX' (pgvector)")
    top_k: int = Field(default=10, ge=1, le=100, description="Number of results to retrieve")
    filter_language: str | None = Field(default=None, max_length=10, description="Optional ISO language code filter")
    filter_dataset_id: uuid.UUID | None = Field(default=None, description="Optional dataset UUID constraint")


class SearchResultItem(BaseModel):
    """Individual ranked search match item."""

    document_id: uuid.UUID
    doc_external_id: str
    score: float
    rank: int
    language_code: str
    script_code: str
    title: str | None = None
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchResponse(BaseModel):
    """Search response schema."""

    query_text: str
    model_id: str
    retrieval_mode: str
    top_k: int
    count: int
    latency_ms: float
    results: list[SearchResultItem]
