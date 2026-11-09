"""Schemas for research experiment configuration, execution, and evaluation reporting."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any
from pydantic import BaseModel, ConfigDict, Field


class ExperimentCreate(BaseModel):
    """Payload to configure a new multilingual evaluation study."""

    name: str = Field(min_length=2, max_length=200, description="Experiment display name")
    description: str | None = Field(default=None, max_length=2000, description="Study description")
    dataset_id: uuid.UUID = Field(description="Target benchmark dataset UUID")
    baseline_language_code: str = Field(default="en", max_length=10, description="Baseline reference language, e.g. 'en'")
    cutoff_k_values: list[int] = Field(default=[1, 5, 10, 20], description="Evaluation cutoff depths")
    similarity_metric: str = Field(default="cosine", pattern=r"^(cosine|dot_product)$", description="Distance metric")


class ExperimentRunRequest(BaseModel):
    """Payload to trigger an experiment run."""

    model_id: str = Field(default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", description="Embedding model identifier or alias")
    retrieval_mode: str = Field(default="EXACT", pattern=r"^(EXACT|VECTOR_INDEX)$", description="Retrieval mode: 'EXACT' or 'VECTOR_INDEX'")
    random_seed: int = Field(default=42, description="Random seed for reproducibility")


class ExperimentRunResponse(BaseModel):
    """Experiment run status and metadata schema."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    experiment_id: uuid.UUID
    model_id: str
    retrieval_mode: str
    status: str
    progress_percent: float
    error_message: str | None = None
    device_used: str
    hardware_metadata: dict[str, Any] | None = None
    total_latency_seconds: float | None = None
    created_at: datetime


class ExperimentResponse(BaseModel):
    """Experiment definition schema."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    description: str | None
    dataset_id: uuid.UUID
    baseline_language_code: str
    cutoff_k_values: list[int]
    similarity_metric: str
    created_at: datetime
    runs: list[ExperimentRunResponse] = Field(default_factory=list)


class PaginatedExperimentResponse(BaseModel):
    """Paginated collection of experiments."""

    total: int
    limit: int
    offset: int
    items: list[ExperimentResponse]


class MetricResultItem(BaseModel):
    """Aggregated evaluation metric item."""

    language_code: str
    script_code: str
    metric_name: str
    metric_value: float
    ci_lower_95: float | None = None
    ci_upper_95: float | None = None
    p_value_vs_baseline: float | None = None
    is_statistically_significant: bool = False


class FairnessDisparityItem(BaseModel):
    """Language fairness disparity summary item."""

    language_code: str
    baseline_language: str
    absolute_performance_gap: float
    gap_ci_lower_95: float | None = None
    gap_ci_upper_95: float | None = None
    relative_disparity_ratio: float
    p_value: float | None = None
    is_statistically_significant: bool = False


class ExperimentEvaluationSummary(BaseModel):
    """Complete evaluation report summary for an experiment."""

    experiment_id: uuid.UUID
    experiment_name: str
    dataset_id: uuid.UUID
    baseline_language: str
    runs: list[ExperimentRunResponse]
    metrics: list[MetricResultItem]
    fairness_disparities: list[FairnessDisparityItem]
    transliteration_penalties: dict[str, float]
