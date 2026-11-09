"""Dataset, Query Group, Query, Document, and Relevance Judgment models."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Integer,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import JSON

from app.db.session import Base


class Dataset(Base):
    """Immutable versioned evaluation dataset."""

    __tablename__ = "datasets"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(150), nullable=False)
    slug: Mapped[str] = mapped_column(String(160), unique=True, nullable=False, index=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    version: Mapped[str] = mapped_column(String(30), nullable=False, default="1.0.0")
    dataset_type: Mapped[str] = mapped_column(
        String(40),
        nullable=False,
        default="VALIDATED_BENCHMARK",
    )  # VALIDATED_BENCHMARK | DEMONSTRATION
    license: Mapped[str] = mapped_column(String(100), default="CC-BY-4.0", nullable=False)
    source_uri: Mapped[str | None] = mapped_column(String(500), nullable=True)
    supported_languages: Mapped[list[str]] = mapped_column(
        JSON,
        nullable=False,
        default=lambda: ["en", "bn", "bn-Latn", "hi", "ar"],
    )
    content_hash_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    is_locked: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    __table_args__ = (
        UniqueConstraint("name", "version", name="uq_dataset_name_version"),
    )

    # Relationships
    query_groups: Mapped[list[QueryGroup]] = relationship(
        "QueryGroup",
        back_populates="dataset",
        cascade="all, delete-orphan",
    )
    queries: Mapped[list[Query]] = relationship(
        "Query",
        back_populates="dataset",
        cascade="all, delete-orphan",
    )
    documents: Mapped[list[Document]] = relationship(
        "Document",
        back_populates="dataset",
        cascade="all, delete-orphan",
    )
    relevance_judgments: Mapped[list[RelevanceJudgment]] = relationship(
        "RelevanceJudgment",
        back_populates="dataset",
        cascade="all, delete-orphan",
    )


class QueryGroup(Base):
    """Invariant semantic intent grouping across language/script variants."""

    __tablename__ = "query_groups"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    dataset_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("datasets.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    canonical_intent: Mapped[str] = mapped_column(Text, nullable=False)
    domain: Mapped[str] = mapped_column(String(100), default="general", nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    # Relationships
    dataset: Mapped[Dataset] = relationship("Dataset", back_populates="query_groups")
    queries: Mapped[list[Query]] = relationship(
        "Query",
        back_populates="query_group",
        cascade="all, delete-orphan",
    )


class Query(Base):
    """Surface realization of a query in a specific language and script."""

    __tablename__ = "queries"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    query_group_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("query_groups.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    dataset_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("datasets.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    query_text: Mapped[str] = mapped_column(Text, nullable=False)
    language_code: Mapped[str] = mapped_column(String(10), nullable=False, index=True)
    script_code: Mapped[str] = mapped_column(String(10), nullable=False, index=True)
    token_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    metadata_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    # Relationships
    query_group: Mapped[QueryGroup] = relationship("QueryGroup", back_populates="queries")
    dataset: Mapped[Dataset] = relationship("Dataset", back_populates="queries")
    relevance_judgments: Mapped[list[RelevanceJudgment]] = relationship(
        "RelevanceJudgment",
        back_populates="query",
        cascade="all, delete-orphan",
    )


class Document(Base):
    """Individual text passage or article in the corpus."""

    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    dataset_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("datasets.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    doc_external_id: Mapped[str] = mapped_column(String(150), nullable=False, index=True)
    title: Mapped[str | None] = mapped_column(Text, nullable=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    language_code: Mapped[str] = mapped_column(String(10), nullable=False, index=True)
    script_code: Mapped[str] = mapped_column(String(10), nullable=False, index=True)
    token_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    metadata_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    # Relationships
    dataset: Mapped[Dataset] = relationship("Dataset", back_populates="documents")
    embeddings: Mapped[list[Any]] = relationship(
        "DocumentEmbedding",
        back_populates="document",
        cascade="all, delete-orphan",
    )
    relevance_judgments: Mapped[list[RelevanceJudgment]] = relationship(
        "RelevanceJudgment",
        back_populates="document",
        cascade="all, delete-orphan",
    )


class RelevanceJudgment(Base):
    """Ground-truth graded relevance ratings between queries and documents."""

    __tablename__ = "relevance_judgments"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    dataset_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("datasets.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    query_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("queries.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    relevance_score: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    judgment_origin: Mapped[str] = mapped_column(
        String(50),
        default="HUMAN_EXPERT",
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    __table_args__ = (
        CheckConstraint("relevance_score >= 0 AND relevance_score <= 3", name="check_relevance_range"),
        UniqueConstraint("query_id", "document_id", name="uq_query_document_judgment"),
    )

    # Relationships
    dataset: Mapped[Dataset] = relationship("Dataset", back_populates="relevance_judgments")
    query: Mapped[Query] = relationship("Query", back_populates="relevance_judgments")
    document: Mapped[Document] = relationship("Document", back_populates="relevance_judgments")
