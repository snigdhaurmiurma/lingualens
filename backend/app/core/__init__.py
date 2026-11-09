"""Domain Core mathematical, information retrieval, and fairness evaluation modules."""

from app.core.metrics import (
    precision_at_k,
    recall_at_k,
    mean_reciprocal_rank,
    discounted_cumulative_gain,
    ideal_discounted_cumulative_gain,
    ndcg_at_k,
    compute_query_metrics,
)
from app.core.fairness import (
    absolute_performance_gap,
    relative_disparity_ratio,
    transliteration_penalty,
    compute_disparity_summary,
)
from app.core.bootstrap import (
    bootstrap_confidence_interval,
    paired_bootstrap_gap_ci,
    paired_wilcoxon_test,
)

__all__ = [
    "precision_at_k",
    "recall_at_k",
    "mean_reciprocal_rank",
    "discounted_cumulative_gain",
    "ideal_discounted_cumulative_gain",
    "ndcg_at_k",
    "compute_query_metrics",
    "absolute_performance_gap",
    "relative_disparity_ratio",
    "transliteration_penalty",
    "compute_disparity_summary",
    "bootstrap_confidence_interval",
    "paired_bootstrap_gap_ci",
    "paired_wilcoxon_test",
]
