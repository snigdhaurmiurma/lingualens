"""Information Retrieval (IR) Evaluation Metrics.

Provides mathematically rigorous, clean-architecture implementations of standard
IR metrics supporting graded relevance judgments:
- Precision@K
- Recall@K
- Mean Reciprocal Rank (MRR)
- Discounted Cumulative Gain (DCG@K)
- Ideal Discounted Cumulative Gain (IDCG@K)
- Normalized Discounted Cumulative Gain (nDCG@K)
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence


def precision_at_k(
    retrieved_ids: Sequence[str],
    relevant_ratings: Mapping[str, int],
    k: int,
    threshold: int = 2,
) -> float:
    """Calculate Precision@K for a single query.

    Precision@K measures the proportion of retrieved documents within the top-K
    ranks that meet or exceed the relevance threshold.

    Formula:
        Precision@K = (Number of relevant documents in top K) / K

    Args:
        retrieved_ids: Ordered sequence of retrieved document IDs (rank 1 to K).
        relevant_ratings: Mapping of document IDs to their ground-truth relevance scores.
        k: Cutoff rank (must be >= 1).
        threshold: Minimum relevance score required to be deemed relevant (default: 2).

    Returns:
        Precision score in the range [0.0, 1.0].
    """
    if k <= 0:
        raise ValueError(f"k must be a positive integer, got {k}")
    if not retrieved_ids:
        return 0.0

    top_k = retrieved_ids[:k]
    relevant_count = sum(
        1 for doc_id in top_k if relevant_ratings.get(doc_id, 0) >= threshold
    )
    return relevant_count / k


def recall_at_k(
    retrieved_ids: Sequence[str],
    relevant_ratings: Mapping[str, int],
    k: int,
    threshold: int = 2,
) -> float:
    """Calculate Recall@K for a single query.

    Recall@K measures what fraction of all relevant documents in the corpus
    were successfully retrieved within the top-K positions.

    Formula:
        Recall@K = (Number of relevant documents in top K) / (Total relevant documents)

    Args:
        retrieved_ids: Ordered sequence of retrieved document IDs (rank 1 to K).
        relevant_ratings: Mapping of document IDs to their ground-truth relevance scores.
        k: Cutoff rank (must be >= 1).
        threshold: Minimum relevance score required to be deemed relevant (default: 2).

    Returns:
        Recall score in the range [0.0, 1.0]. If no relevant documents exist in the
        ground truth, returns 0.0.
    """
    if k <= 0:
        raise ValueError(f"k must be a positive integer, got {k}")
    if not retrieved_ids:
        return 0.0

    total_relevant = sum(
        1 for score in relevant_ratings.values() if score >= threshold
    )
    if total_relevant == 0:
        return 0.0

    top_k = retrieved_ids[:k]
    relevant_retrieved = sum(
        1 for doc_id in top_k if relevant_ratings.get(doc_id, 0) >= threshold
    )
    return relevant_retrieved / total_relevant


def mean_reciprocal_rank(
    retrieved_ids: Sequence[str],
    relevant_ratings: Mapping[str, int],
    threshold: int = 2,
) -> float:
    """Calculate Reciprocal Rank (RR) for a single query.

    Reciprocal Rank is the inverse of the 1-based rank position of the first
    relevant document retrieved.

    Formula:
        RR = 1 / rank_first_relevant (or 0.0 if no relevant document was retrieved)

    Args:
        retrieved_ids: Ordered sequence of retrieved document IDs (rank 1 to N).
        relevant_ratings: Mapping of document IDs to their ground-truth relevance scores.
        threshold: Minimum relevance score required to be deemed relevant (default: 2).

    Returns:
        Reciprocal Rank in the range [0.0, 1.0].
    """
    for rank_idx, doc_id in enumerate(retrieved_ids, start=1):
        if relevant_ratings.get(doc_id, 0) >= threshold:
            return 1.0 / rank_idx
    return 0.0


def discounted_cumulative_gain(
    retrieved_ids: Sequence[str],
    relevant_ratings: Mapping[str, int],
    k: int,
) -> float:
    """Calculate Discounted Cumulative Gain (DCG@K) using graded relevance.

    Uses the standard exponential gain formulation favored in modern IR:
        DCG@K = sum_{i=1}^{min(k, |R|)} (2^{r_i} - 1) / log2(i + 1)

    where r_i is the graded relevance rating (0 to 3) of the document at rank i.

    Args:
        retrieved_ids: Ordered sequence of retrieved document IDs.
        relevant_ratings: Mapping of document IDs to their graded relevance scores.
        k: Cutoff rank (must be >= 1).

    Returns:
        DCG@K value (non-negative float).
    """
    if k <= 0:
        raise ValueError(f"k must be a positive integer, got {k}")

    dcg = 0.0
    for rank_idx, doc_id in enumerate(retrieved_ids[:k], start=1):
        rel = relevant_ratings.get(doc_id, 0)
        gain = (2.0 ** rel) - 1.0
        discount = math.log2(rank_idx + 1)
        dcg += gain / discount
    return dcg


def ideal_discounted_cumulative_gain(
    relevant_ratings: Mapping[str, int],
    k: int,
) -> float:
    """Calculate Ideal Discounted Cumulative Gain (IDCG@K).

    Sorts all available relevance ratings in descending order and calculates
    the theoretical maximum DCG attainable for top-K documents.

    Args:
        relevant_ratings: Mapping of document IDs to their graded relevance scores.
        k: Cutoff rank (must be >= 1).

    Returns:
        IDCG@K value (non-negative float).
    """
    if k <= 0:
        raise ValueError(f"k must be a positive integer, got {k}")

    # Sort available ratings descending
    sorted_ratings = sorted(relevant_ratings.values(), reverse=True)
    idcg = 0.0
    for rank_idx, rel in enumerate(sorted_ratings[:k], start=1):
        if rel <= 0:
            break
        gain = (2.0 ** rel) - 1.0
        discount = math.log2(rank_idx + 1)
        idcg += gain / discount
    return idcg


def ndcg_at_k(
    retrieved_ids: Sequence[str],
    relevant_ratings: Mapping[str, int],
    k: int,
) -> float:
    """Calculate Normalized Discounted Cumulative Gain (nDCG@K).

    Formula:
        nDCG@K = DCG@K / IDCG@K

    If IDCG@K is 0 (i.e., no relevant documents exist in the ground truth),
    returns 0.0 to prevent division by zero.

    Args:
        retrieved_ids: Ordered sequence of retrieved document IDs.
        relevant_ratings: Mapping of document IDs to their graded relevance scores.
        k: Cutoff rank (must be >= 1).

    Returns:
        nDCG score in the range [0.0, 1.0].
    """
    idcg = ideal_discounted_cumulative_gain(relevant_ratings, k)
    if idcg <= 0.0:
        return 0.0
    dcg = discounted_cumulative_gain(retrieved_ids, relevant_ratings, k)
    return min(1.0, max(0.0, dcg / idcg))


def compute_query_metrics(
    retrieved_ids: Sequence[str],
    relevant_ratings: Mapping[str, int],
    cutoff_k_values: Sequence[int] = (1, 5, 10, 20),
    threshold: int = 2,
) -> dict[str, float]:
    """Compute a standard suite of IR metrics for a single query evaluation.

    Args:
        retrieved_ids: Ordered sequence of retrieved document IDs.
        relevant_ratings: Mapping of document IDs to graded scores.
        cutoff_k_values: List of k cutoffs to compute (e.g. [1, 5, 10, 20]).
        threshold: Binary relevance threshold for Precision/Recall/MRR.

    Returns:
        Dictionary mapping metric names (e.g. 'ndcg@10', 'mrr') to computed floats.
    """
    metrics: dict[str, float] = {
        "mrr": mean_reciprocal_rank(retrieved_ids, relevant_ratings, threshold)
    }

    for k in cutoff_k_values:
        metrics[f"precision@{k}"] = precision_at_k(retrieved_ids, relevant_ratings, k, threshold)
        metrics[f"recall@{k}"] = recall_at_k(retrieved_ids, relevant_ratings, k, threshold)
        metrics[f"ndcg@{k}"] = ndcg_at_k(retrieved_ids, relevant_ratings, k)

    return metrics
