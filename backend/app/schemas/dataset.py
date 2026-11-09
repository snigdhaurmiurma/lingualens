"""Schemas for Dataset, Document, and Query management."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any
from pydantic import BaseModel, Field, field_validator


class DatasetCreate(BaseModel):
    """Payload to create a new evaluation dataset."""

    name: str = Field(min_length=2, max_length=150, description="Dataset display name")
    slug: str = Field(min_length=2, max_length=160, pattern=r"^[a-z0-9-_]+$", description="URL-safe unique identifier")
    version: str = Field(default="1.0.0", max_length=30, description="Semantic version string")
    description: str | None = Field(default=None, max_length=2000, description="Dataset summary")
    supported_languages: list[str] = Field(
        default=["en", "bn", "bn-Latn", "hi", "ar"],
        description="Supported ISO language and script codes",
    )
    dataset_type: str = Field(
        default="VALIDATED_BENCHMARK",
        pattern=r"^(VALIDATED_BENCHMARK|DEMONSTRATION)$",
        description="Dataset verification category",
    )
    source_uri: str | None = Field(default=None, max_length=500, description="Provenance link or DOI")


class DatasetResponse(BaseModel):
    """Dataset response schema."""

    id: uuid.UUID
    name: str
    slug: str
    version: str
    description: str | None
    dataset_type: str
    license: str
    supported_languages: list[str]
    content_hash_sha256: str
    is_locked: bool
    document_count: int = 0
    query_count: int = 0
    query_group_count: int = 0
    created_at: datetime


class DocumentCreate(BaseModel):
    """Payload to add a document to a dataset."""

    doc_external_id: str = Field(min_length=1, max_length=150, description="Source or external ID")
    content: str = Field(min_length=1, description="Document body text")
    language_code: str = Field(min_length=2, max_length=10, description="ISO language code, e.g. 'bn', 'en'")
    script_code: str = Field(default="Latn", min_length=4, max_length=10, description="ISO script code, e.g. 'Beng'")
    title: str | None = Field(default=None, max_length=500, description="Optional title")
    metadata: dict[str, Any] | None = Field(default=None, description="Arbitrary metadata dictionary")


class DocumentResponse(BaseModel):
    """Document response schema."""

    id: uuid.UUID
    dataset_id: uuid.UUID
    doc_external_id: str
    title: str | None
    content: str
    language_code: str
    script_code: str
    metadata: dict[str, Any] | None
    created_at: datetime


class QueryGroupCreate(BaseModel):
    """Payload to create an invariant semantic intent group."""

    canonical_intent: str = Field(min_length=2, description="Canonical representation of query intent")
    domain: str = Field(default="general", max_length=100, description="Domain category, e.g. healthcare")


class QueryGroupResponse(BaseModel):
    """Query group response schema."""

    id: uuid.UUID
    dataset_id: uuid.UUID
    canonical_intent: str
    domain: str
    created_at: datetime


class QueryCreate(BaseModel):
    """Payload to add a language/script query realization."""

    query_group_id: uuid.UUID = Field(description="Parent QueryGroup ID representing the semantic intent")
    query_text: str = Field(min_length=1, description="Localized query string")
    language_code: str = Field(min_length=2, max_length=10, description="ISO language code")
    script_code: str = Field(default="Latn", min_length=4, max_length=10, description="ISO script code")
    metadata: dict[str, Any] | None = Field(default=None, description="Optional query metadata")


class QueryResponse(BaseModel):
    """Query response schema."""

    id: uuid.UUID
    query_group_id: uuid.UUID
    dataset_id: uuid.UUID
    query_text: str
    language_code: str
    script_code: str
    metadata: dict[str, Any] | None
    created_at: datetime
