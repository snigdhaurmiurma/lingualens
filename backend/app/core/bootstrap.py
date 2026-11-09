"""Statistical Resampling and Hypothesis Testing.

Provides statistical significance testing and empirical confidence intervals:
- Non-parametric Percentile Bootstrap Resampling
- Paired Disparity Bootstrap Confidence Intervals
- Paired Wilcoxon Signed-Rank Test (Scipy)
"""

from __future__ import annotations

from typing import Sequence
import numpy as np
from scipy import stats


def bootstrap_confidence_interval(
    scores: Sequence[float] | np.ndarray,
    num_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_seed: int | None = 42,
) -> tuple[float, float, float]:
    """Compute empirical non-parametric bootstrap confidence intervals for a metric.

    Because IR metrics are non-Gaussian and bounded in [0, 1], normal distribution
    assumptions fail. The percentile bootstrap draws B resamples with replacement
    to empirically estimate the sampling distribution of the mean.

    Args:
        scores: Sequence of metric values for queries.
        num_bootstrap: Number of bootstrap iterations (default: 1000).
        confidence_level: Desired coverage probability (default: 0.95 for 95% CI).
        random_seed: Seed for pseudo-random number generator for reproducible runs.

    Returns:
        Tuple of (empirical_mean, ci_lower, ci_upper).
    """
    arr = np.asarray(scores, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return 0.0, 0.0, 0.0
    if n == 1:
        val = float(arr[0])
        return val, val, val

    rng = np.random.default_rng(random_seed)
    # Generate B bootstrap sample indices of shape (num_bootstrap, n)
    indices = rng.integers(0, n, size=(num_bootstrap, n))
    bootstrap_means = np.mean(arr[indices], axis=1)

    alpha = 1.0 - confidence_level
    lower_percentile = (alpha / 2.0) * 100.0
    upper_percentile = (1.0 - alpha / 2.0) * 100.0

    ci_lower = float(np.percentile(bootstrap_means, lower_percentile))
    ci_upper = float(np.percentile(bootstrap_means, upper_percentile))
    mean_val = float(np.mean(arr))

    return mean_val, ci_lower, ci_upper


def paired_bootstrap_gap_ci(
    baseline_scores: Sequence[float] | np.ndarray,
    target_scores: Sequence[float] | np.ndarray,
    num_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_seed: int | None = 42,
) -> tuple[float, float, float]:
    """Compute bootstrap confidence interval for the paired performance gap.

    Resamples paired query indices (g_1, ..., g_n) with replacement to construct
    the sampling distribution of Delta_abs = mean(baseline) - mean(target).

    Args:
        baseline_scores: Query metric values for the baseline language.
        target_scores: Paired query metric values for the target language.
        num_bootstrap: Number of bootstrap resamples (default: 1000).
        confidence_level: Confidence level (default: 0.95).
        random_seed: Random seed for deterministic reproducibility.

    Returns:
        Tuple of (mean_gap, ci_lower, ci_upper).
    """
    base = np.asarray(baseline_scores, dtype=np.float64)
    target = np.asarray(target_scores, dtype=np.float64)

    if len(base) != len(target):
        raise ValueError(
            f"Paired bootstrap requires equal sample lengths: "
            f"{len(base)} baseline vs {len(target)} target"
        )
    n = len(base)
    if n == 0:
        return 0.0, 0.0, 0.0

    # Differences per query group
    diffs = base - target
    if n == 1:
        val = float(diffs[0])
        return val, val, val

    rng = np.random.default_rng(random_seed)
    indices = rng.integers(0, n, size=(num_bootstrap, n))
    bootstrap_gap_means = np.mean(diffs[indices], axis=1)

    alpha = 1.0 - confidence_level
    lower_p = (alpha / 2.0) * 100.0
    upper_p = (1.0 - alpha / 2.0) * 100.0

    ci_lower = float(np.percentile(bootstrap_gap_means, lower_p))
    ci_upper = float(np.percentile(bootstrap_gap_means, upper_p))
    mean_gap = float(np.mean(diffs))

    return mean_gap, ci_lower, ci_upper


def paired_wilcoxon_test(
    baseline_scores: Sequence[float] | np.ndarray,
    target_scores: Sequence[float] | np.ndarray,
    significance_alpha: float = 0.05,
) -> tuple[float, float, bool]:
    """Perform a Paired Wilcoxon Signed-Rank Test across paired query metrics.

    Tests the null hypothesis H0 that the distribution of metric differences
    between baseline and target languages is symmetric about zero (no disparity).

    Args:
        baseline_scores: Metric values for baseline language.
        target_scores: Paired metric values for target language.
        significance_alpha: p-value significance threshold (default: 0.05).

    Returns:
        Tuple of (test_statistic, p_value, is_statistically_significant).
    """
    base = np.asarray(baseline_scores, dtype=np.float64)
    target = np.asarray(target_scores, dtype=np.float64)

    if len(base) != len(target):
        raise ValueError("Paired query length mismatch for Wilcoxon test.")
    if len(base) == 0:
        return 0.0, 1.0, False

    diffs = base - target

    # Edge case: If all paired differences are identical or zero, test is non-significant
    if np.all(diffs == 0.0):
        return 0.0, 1.0, False

    try:
        # Two-sided Wilcoxon signed-rank test
        result = stats.wilcoxon(base, target, alternative="two-sided")
        stat = float(result.statistic)
        p_val = float(result.pvalue)
        is_sig = p_val < significance_alpha
        return stat, p_val, is_sig
    except ValueError:
        # Handled in case of insufficient non-zero differences
        return 0.0, 1.0, False
