"""Document embedding entity supporting dual-mode storage (PostgreSQL pgvector / SQLite JSON fallback)."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

import numpy as np
from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, TypeDecorator
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import JSON

from app.db.session import Base


class VectorType(TypeDecorator):
    """Adaptive vector column supporting native pgvector on PostgreSQL and JSON arrays on SQLite."""

    impl = JSON
    cache_ok = True

    def __init__(self, dim: int = 384, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.dim = dim

    def load_dialect_impl(self, dialect: Any) -> Any:
        if dialect.name == "postgresql":
            try:
                from pgvector.sqlalchemy import Vector

                return dialect.type_descriptor(Vector(self.dim))
            except ImportError:
                return dialect.type_descriptor(JSONB())
        return dialect.type_descriptor(JSON())

    def process_bind_param(self, value: Any, dialect: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, np.ndarray):
            return value.tolist()
        return value

    def process_result_value(self, value: Any, dialect: Any) -> np.ndarray | None:
        if value is None:
            return None
        if hasattr(value, "to_numpy"):
            return value.to_numpy().astype(np.float32)
        if isinstance(value, np.ndarray):
            return value.astype(np.float32)
        return np.array(value, dtype=np.float32)


class DocumentEmbedding(Base):
    """Stores high-dimensional dense vector representations for documents per model."""

    __tablename__ = "document_embeddings"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    model_id: Mapped[str] = mapped_column(
        String(200),
        nullable=False,
        index=True,
    )
    embedding: Mapped[np.ndarray] = mapped_column(
        VectorType(dim=384),
        nullable=False,
    )
    dimension: Mapped[int] = mapped_column(Integer, default=384, nullable=False)
    is_normalized: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    # Relationships
    document: Mapped[Any] = relationship("Document", back_populates="embeddings")
