"""Unit tests for Information Retrieval (IR) evaluation metrics."""

import math
import pytest
from app.core.metrics import (
    precision_at_k,
    recall_at_k,
    mean_reciprocal_rank,
    discounted_cumulative_gain,
    ideal_discounted_cumulative_gain,
    ndcg_at_k,
    compute_query_metrics,
)


@pytest.fixture
def sample_retrieval_data():
    """Provides a known, analytically verified retrieval run."""
    retrieved_ids = ["doc_A", "doc_B", "doc_C", "doc_D", "doc_E"]
    # Graded ratings: doc_A is 3, doc_B is 1, doc_C is 2, doc_D is 0, doc_E is 3
    # Relevant at threshold >= 2: doc_A, doc_C, doc_E (total 3 relevant documents)
    relevant_ratings = {
        "doc_A": 3,
        "doc_B": 1,
        "doc_C": 2,
        "doc_D": 0,
        "doc_E": 3,
    }
    return retrieved_ids, relevant_ratings


def test_precision_at_k(sample_retrieval_data):
    retrieved, ratings = sample_retrieval_data

    # Top 1: ["doc_A"] -> doc_A is 3 (>=2) -> 1/1 = 1.0
    assert precision_at_k(retrieved, ratings, k=1, threshold=2) == 1.0

    # Top 2: ["doc_A", "doc_B"] -> doc_A is 3, doc_B is 1 (<2) -> 1/2 = 0.5
    assert precision_at_k(retrieved, ratings, k=2, threshold=2) == 0.5

    # Top 3: ["doc_A", "doc_B", "doc_C"] -> doc_A (3), doc_C (2) -> 2/3
    assert precision_at_k(retrieved, ratings, k=3, threshold=2) == pytest.approx(2 / 3)

    # Top 5: ["doc_A", "doc_B", "doc_C", "doc_D", "doc_E"] -> 3/5 = 0.6
    assert precision_at_k(retrieved, ratings, k=5, threshold=2) == 0.6


def test_recall_at_k(sample_retrieval_data):
    retrieved, ratings = sample_retrieval_data
    # Total relevant in ground truth with score >= 2: doc_A, doc_C, doc_E (total = 3)

    # Top 1: retrieved doc_A (1 relevant) -> 1/3
    assert recall_at_k(retrieved, ratings, k=1, threshold=2) == pytest.approx(1 / 3)

    # Top 3: retrieved doc_A, doc_C (2 relevant) -> 2/3
    assert recall_at_k(retrieved, ratings, k=3, threshold=2) == pytest.approx(2 / 3)

    # Top 5: retrieved doc_A, doc_C, doc_E (3 relevant) -> 3/3 = 1.0
    assert recall_at_k(retrieved, ratings, k=5, threshold=2) == 1.0


def test_mean_reciprocal_rank(sample_retrieval_data):
    retrieved, ratings = sample_retrieval_data

    # First relevant doc is doc_A at rank 1 -> RR = 1/1 = 1.0
    assert mean_reciprocal_rank(retrieved, ratings, threshold=2) == 1.0

    # Test when first relevant doc is at rank 3
    shifted_retrieved = ["doc_D", "doc_B", "doc_C", "doc_A"]
    # doc_D is 0, doc_B is 1, doc_C is 2 (first relevant at rank 3) -> RR = 1/3
    assert mean_reciprocal_rank(shifted_retrieved, ratings, threshold=2) == pytest.approx(1 / 3)

    # Test when no relevant documents are retrieved
    irrelevant_only = ["doc_D", "doc_B"]
    assert mean_reciprocal_rank(irrelevant_only, ratings, threshold=2) == 0.0


def test_dcg_and_idcg_analytical_calculation():
    """Verify DCG and IDCG against manual hand-calculated values."""
    retrieved = ["doc_1", "doc_2", "doc_3"]
    ratings = {
        "doc_1": 3,
        "doc_2": 1,
        "doc_3": 2,
    }

    # Manual calculation:
    # Rank 1: (2^3 - 1) / log2(2) = 7 / 1.0 = 7.0
    # Rank 2: (2^1 - 1) / log2(3) = 1 / 1.58496250072 = 0.63092975357
    # Rank 3: (2^2 - 1) / log2(4) = 3 / 2.0 = 1.5
    expected_dcg_3 = 7.0 + (1.0 / math.log2(3)) + 1.5
    calculated_dcg = discounted_cumulative_gain(retrieved, ratings, k=3)
    assert calculated_dcg == pytest.approx(expected_dcg_3, rel=1e-6)

    # For IDCG@3:
    # Ideal descending ratings: [3, 2, 1]
    # Rank 1: 3 -> 7.0
    # Rank 2: 2 -> (2^2 - 1) / log2(3) = 3 / log2(3) = 1.8927892607
    # Rank 3: 1 -> (2^1 - 1) / log2(4) = 1 / 2.0 = 0.5
    expected_idcg_3 = 7.0 + (3.0 / math.log2(3)) + 0.5
    calculated_idcg = ideal_discounted_cumulative_gain(ratings, k=3)
    assert calculated_idcg == pytest.approx(expected_idcg_3, rel=1e-6)

    # nDCG@3 = DCG@3 / IDCG@3
    expected_ndcg = expected_dcg_3 / expected_idcg_3
    calculated_ndcg = ndcg_at_k(retrieved, ratings, k=3)
    assert calculated_ndcg == pytest.approx(expected_ndcg, rel=1e-6)


def test_ndcg_perfect_ranking():
    """When ranking is ideal, nDCG must be exactly 1.0."""
    retrieved = ["d_best", "d_good", "d_okay"]
    ratings = {"d_best": 3, "d_good": 2, "d_okay": 1}

    assert ndcg_at_k(retrieved, ratings, k=1) == 1.0
    assert ndcg_at_k(retrieved, ratings, k=2) == 1.0
    assert ndcg_at_k(retrieved, ratings, k=3) == 1.0


def test_metrics_edge_cases():
    # Empty inputs
    assert precision_at_k([], {}, k=5) == 0.0
    assert recall_at_k([], {}, k=5) == 0.0
    assert mean_reciprocal_rank([], {}) == 0.0
    assert ndcg_at_k([], {}, k=5) == 0.0

    # No relevant documents in ground truth
    all_zero_ratings = {"doc_1": 0, "doc_2": 0}
    assert recall_at_k(["doc_1"], all_zero_ratings, k=1) == 0.0
    assert ndcg_at_k(["doc_1"], all_zero_ratings, k=1) == 0.0

    # Invalid k values
    with pytest.raises(ValueError):
        precision_at_k(["doc_1"], {"doc_1": 2}, k=0)
    with pytest.raises(ValueError):
        discounted_cumulative_gain(["doc_1"], {"doc_1": 2}, k=-1)


def test_compute_query_metrics_suite():
    retrieved = ["doc_A", "doc_B"]
    ratings = {"doc_A": 3, "doc_B": 0}
    results = compute_query_metrics(retrieved, ratings, cutoff_k_values=[1, 5])

    assert "mrr" in results
    assert "precision@1" in results
    assert "recall@1" in results
    assert "ndcg@1" in results
    assert "precision@5" in results
    assert results["mrr"] == 1.0
    assert results["precision@1"] == 1.0
    assert results["ndcg@1"] == 1.0
