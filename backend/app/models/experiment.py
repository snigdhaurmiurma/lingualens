"""Experiment, ExperimentRun, RetrievedDocument, and EvaluationResult models."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    SmallInteger,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import JSON

from app.db.session import Base


class Experiment(Base):
    """High-level evaluation study comparing models or languages."""

    __tablename__ = "experiments"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    dataset_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("datasets.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    baseline_language_code: Mapped[str] = mapped_column(String(10), default="en", nullable=False)
    cutoff_k_values: Mapped[list[int]] = mapped_column(
        JSON,
        default=lambda: [1, 5, 10, 20],
        nullable=False,
    )
    similarity_metric: Mapped[str] = mapped_column(String(50), default="cosine", nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    # Relationships
    runs: Mapped[list[ExperimentRun]] = relationship(
        "ExperimentRun",
        back_populates="experiment",
        cascade="all, delete-orphan",
    )


class ExperimentRun(Base):
    """Specific model run under an experiment."""

    __tablename__ = "experiment_runs"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    experiment_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("experiments.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    model_id: Mapped[str] = mapped_column(String(200), nullable=False, index=True)
    retrieval_mode: Mapped[str] = mapped_column(
        String(50),
        default="EXACT",
        nullable=False,
    )  # EXACT | VECTOR_INDEX
    status: Mapped[str] = mapped_column(
        String(30),
        default="PENDING",
        nullable=False,
    )  # PENDING | RUNNING | COMPLETED | FAILED
    progress_percent: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    random_seed: Mapped[int] = mapped_column(Integer, default=42, nullable=False)
    device_used: Mapped[str] = mapped_column(String(50), default="cpu", nullable=False)
    hardware_metadata: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    total_latency_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    # Relationships
    experiment: Mapped[Experiment] = relationship("Experiment", back_populates="runs")
    retrieved_documents: Mapped[list[RetrievedDocument]] = relationship(
        "RetrievedDocument",
        back_populates="run",
        cascade="all, delete-orphan",
    )
    evaluation_results: Mapped[list[EvaluationResult]] = relationship(
        "EvaluationResult",
        back_populates="run",
        cascade="all, delete-orphan",
    )


class RetrievedDocument(Base):
    """Detailed top-K SERP audit trail for an individual query execution."""

    __tablename__ = "retrieved_documents"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    experiment_run_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("experiment_runs.id", ondelete="CASCADE"),
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
    rank_position: Mapped[int] = mapped_column(Integer, nullable=False)
    similarity_score: Mapped[float] = mapped_column(Float, nullable=False)
    assigned_relevance: Mapped[int] = mapped_column(SmallInteger, default=0, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    # Relationships
    run: Mapped[ExperimentRun] = relationship("ExperimentRun", back_populates="retrieved_documents")


class EvaluationResult(Base):
    """Aggregated evaluation and fairness metrics per language/script partition."""

    __tablename__ = "evaluation_results"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    experiment_run_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("experiment_runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    language_code: Mapped[str] = mapped_column(String(10), nullable=False, index=True)
    script_code: Mapped[str] = mapped_column(String(10), nullable=False, index=True)
    metric_name: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    metric_value: Mapped[float] = mapped_column(Float, nullable=False)
    ci_lower_95: Mapped[float | None] = mapped_column(Float, nullable=True)
    ci_upper_95: Mapped[float | None] = mapped_column(Float, nullable=True)
    p_value_vs_baseline: Mapped[float | None] = mapped_column(Float, nullable=True)
    is_statistically_significant: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    # Relationships
    run: Mapped[ExperimentRun] = relationship("ExperimentRun", back_populates="evaluation_results")
