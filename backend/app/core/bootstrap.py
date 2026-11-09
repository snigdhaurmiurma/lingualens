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
    zero_method: str = "wilcox",
) -> tuple[float, float, bool]:
    """Perform a Paired Wilcoxon Signed-Rank Test across paired query metrics.

    Statistical Hypotheses:
        - Null Hypothesis (H0): The distribution of paired differences
          D_i = M(q_{i, baseline}) - M(q_{i, target}) is symmetric about zero (no disparity).
        - Alternative Hypothesis (H1): The distribution of differences is shifted away
          from zero (two-sided).

    Core Statistical Assumptions:
        1. Paired Observations: Queries must be strictly paired (belonging to the
           identical semantic query group Q_G representing equivalent intent).
        2. Independent Pairs: Each query group pair (x_i, y_i) is sampled independently
           from other query groups.
        3. Measurement Scale: Differences D_i are ordinal or continuous (e.g. nDCG deltas).
        4. Symmetry under H0: The distribution of differences is assumed symmetric under H0.
           Critically, normality is NOT assumed, making this test appropriate for bounded
           IR metrics.

    Interpretation of p-value:
        The p-value is the probability of observing a test statistic at least as
        extreme as the one calculated, assuming H0 is true.
        - If p < alpha (default: 0.05), we reject H0 at significance level alpha,
          concluding that the observed cross-lingual gap is unlikely to be an artifact
          of sampling noise under the null model.
        - Caution: A p-value is NOT the probability that the alternative hypothesis is true.

    Args:
        baseline_scores: Metric values for baseline language (length N).
        target_scores: Paired metric values for target language (length N).
        significance_alpha: Significance threshold alpha (default: 0.05).
        zero_method: How differences equal to zero are treated ('wilcox', 'pratt', 'zsplit').
                     'wilcox' (default) discards zero differences per standard methodology.

    Returns:
        Tuple of (test_statistic, p_value, is_statistically_significant).
    """
    base = np.asarray(baseline_scores, dtype=np.float64)
    target = np.asarray(target_scores, dtype=np.float64)

    if len(base) != len(target):
        raise ValueError(
            f"Paired Wilcoxon test requires equal sample lengths: "
            f"{len(base)} baseline vs {len(target)} target"
        )
    if len(base) == 0:
        return 0.0, 1.0, False

    diffs = base - target

    # If all differences are zero, H0 cannot be rejected
    non_zero_diffs = diffs[diffs != 0.0]
    if len(non_zero_diffs) == 0:
        return 0.0, 1.0, False

    try:
        # Two-sided Wilcoxon signed-rank test
        result = stats.wilcoxon(base, target, alternative="two-sided", zero_method=zero_method)
        stat = float(result.statistic)
        p_val = float(result.pvalue)
        is_sig = p_val < significance_alpha
        return stat, p_val, is_sig
    except ValueError:
        # Occurs if sample size after zero removal is too small for the selected method
        return 0.0, 1.0, False
