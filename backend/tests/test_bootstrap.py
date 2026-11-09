"""Unit tests for Bootstrap Resampling and Hypothesis Testing."""

import numpy as np
import pytest
from app.core.bootstrap import (
    bootstrap_confidence_interval,
    paired_bootstrap_gap_ci,
    paired_wilcoxon_test,
)


def test_bootstrap_confidence_interval_reproducibility():
    scores = [0.82, 0.75, 0.90, 0.68, 0.85, 0.79, 0.88, 0.72]

    # Deterministic seed test
    mean_1, low_1, high_1 = bootstrap_confidence_interval(scores, random_seed=42)
    mean_2, low_2, high_2 = bootstrap_confidence_interval(scores, random_seed=42)

    assert mean_1 == mean_2
    assert low_1 == low_2
    assert high_1 == high_2

    # Bounds validity: lower <= mean <= upper
    assert low_1 <= mean_1 <= high_1


def test_paired_bootstrap_gap_ci():
    # Baseline is systematically higher than target by approximately 0.3
    baseline = [0.85, 0.90, 0.78, 0.88, 0.92, 0.80]
    target = [0.55, 0.60, 0.48, 0.58, 0.62, 0.50]

    mean_gap, ci_low, ci_high = paired_bootstrap_gap_ci(
        baseline, target, num_bootstrap=1000, random_seed=123
    )

    # Expected mean gap is exactly 0.30
    assert mean_gap == pytest.approx(0.30)
    # The 95% confidence interval must encompass 0.30
    assert ci_low <= 0.30 <= ci_high
    assert ci_low > 0.20
    assert ci_high < 0.40


def test_paired_wilcoxon_test_statistically_significant():
    # Clear, consistent advantage for baseline across 10 query pairs
    baseline = [0.85, 0.88, 0.90, 0.82, 0.79, 0.91, 0.86, 0.84, 0.87, 0.89]
    target = [0.42, 0.45, 0.48, 0.40, 0.38, 0.49, 0.43, 0.41, 0.44, 0.46]

    stat, p_val, is_sig = paired_wilcoxon_test(baseline, target, significance_alpha=0.05)

    assert is_sig is True
    assert p_val < 0.01  # Strong statistical significance


def test_paired_wilcoxon_test_identical_distributions():
    # Identical values have 0 difference everywhere
    data = [0.75, 0.80, 0.85, 0.90]
    stat, p_val, is_sig = paired_wilcoxon_test(data, data)

    assert stat == 0.0
    assert p_val == 1.0
    assert is_sig is False


def test_bootstrap_edge_cases():
    # Single element list
    single_val = [0.75]
    mean, low, high = bootstrap_confidence_interval(single_val)
    assert mean == 0.75
    assert low == 0.75
    assert high == 0.75

    # Empty list
    mean_empty, low_empty, high_empty = bootstrap_confidence_interval([])
    assert mean_empty == 0.0
    assert low_empty == 0.0
    assert high_empty == 0.0

    # Length mismatch in paired test
    with pytest.raises(ValueError):
        paired_bootstrap_gap_ci([0.8, 0.9], [0.5])
