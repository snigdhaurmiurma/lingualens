"""Cross-Lingual Fairness and Disparity Metrics.

Provides statistical functions to quantify retrieval performance disparity
across language variants and script systems:
- Absolute Performance Gap (Delta_abs)
- Relative Disparity Ratio (D_rel)
- Transliteration Penalty (Phi_trans)
- Disparity Matrix and Summary
"""

from __future__ import annotations

import statistics
from typing import Mapping, Sequence


def absolute_performance_gap(
    baseline_scores: Sequence[float],
    target_scores: Sequence[float],
) -> float:
    """Calculate the Absolute Performance Gap (Delta_abs) between two languages.

    Formula:
        Delta_abs = mean(baseline_scores) - mean(target_scores)

    Interpretation:
        - Delta_abs > 0: Target language performs worse than the baseline.
        - Delta_abs == 0: Exact parity between languages.
        - Delta_abs < 0: Target language outperforms the baseline.

    Args:
        baseline_scores: Metric values (e.g., nDCG@10) for the baseline language (e.g. English).
        target_scores: Paired metric values for the target language (e.g. Bengali).

    Returns:
        Absolute difference between the average performance scores.
    """
    if len(baseline_scores) != len(target_scores):
        raise ValueError(
            f"Paired query length mismatch: "
            f"got {len(baseline_scores)} baseline vs {len(target_scores)} target"
        )
    if not baseline_scores:
        return 0.0

    mean_baseline = statistics.fmean(baseline_scores)
    mean_target = statistics.fmean(target_scores)
    return mean_baseline - mean_target


def relative_disparity_ratio(
    baseline_scores: Sequence[float],
    target_scores: Sequence[float],
    epsilon: float = 1e-9,
) -> float:
    """Calculate the Relative Disparity Ratio (D_rel).

    Formula:
        D_rel = mean(target_scores) / (mean(baseline_scores) + epsilon)

    Interpretation:
        - D_rel = 1.0: Perfect parity with the baseline.
        - D_rel < 1.0: Underperformance relative to baseline (e.g., 0.65 = 65% of baseline performance).
        - D_rel > 1.0: Outperformance relative to baseline.

    Args:
        baseline_scores: Metric scores for baseline language.
        target_scores: Paired metric scores for target language.
        epsilon: Small smoothing constant to prevent division by zero.

    Returns:
        Ratio of target performance relative to baseline.
    """
    if len(baseline_scores) != len(target_scores):
        raise ValueError("Paired query length mismatch.")
    if not baseline_scores:
        return 1.0

    mean_target = statistics.fmean(target_scores)
    mean_baseline = statistics.fmean(baseline_scores)

    if mean_baseline == 0.0 and mean_target == 0.0:
        return 1.0

    return mean_target / (mean_baseline + epsilon)


def transliteration_penalty(
    native_scores: Sequence[float],
    transliterated_scores: Sequence[float],
) -> float:
    """Calculate the Transliteration Penalty (Phi_trans).

    Measures the specific retrieval collapse that occurs when a user formulates
    their query in Latin script (e.g. Banglish, bn-Latn) rather than the language's
    native script (e.g. Bengali, bn-Beng).

    Formula:
        Phi_trans = mean(native_scores) - mean(transliterated_scores)

    Args:
        native_scores: Metric scores for native script queries (e.g. 'মৌসুমি ফ্লুর চিকিৎসা').
        transliterated_scores: Metric scores for transliterated queries (e.g. 'mousumi flur chikitsa').

    Returns:
        Drop in retrieval effectiveness attributable purely to script transliteration.
    """
    return absolute_performance_gap(native_scores, transliterated_scores)


def compute_disparity_summary(
    language_metrics: Mapping[str, Sequence[float]],
    baseline_language: str = "en",
) -> dict[str, dict[str, float]]:
    """Generate a comprehensive cross-lingual disparity audit across all languages.

    Args:
        language_metrics: Dictionary mapping language code (e.g. 'en', 'bn', 'bn-Latn')
                          to a list of per-query metric values.
        baseline_language: Language code to serve as the reference baseline (default: 'en').

    Returns:
        Dictionary mapping each language to its statistical profile:
        - 'mean': Average metric score.
        - 'std': Standard deviation of metric score across queries.
        - 'absolute_gap': Performance gap relative to baseline (Delta_abs).
        - 'relative_disparity': Performance ratio relative to baseline (D_rel).
    """
    if baseline_language not in language_metrics:
        raise ValueError(
            f"Baseline language '{baseline_language}' not found in provided metrics. "
            f"Available languages: {list(language_metrics.keys())}"
        )

    baseline_scores = language_metrics[baseline_language]
    summary: dict[str, dict[str, float]] = {}

    for lang_code, scores in language_metrics.items():
        count = len(scores)
        mean_val = statistics.fmean(scores) if count > 0 else 0.0
        std_val = statistics.stdev(scores) if count > 1 else 0.0

        gap = absolute_performance_gap(baseline_scores, scores)
        rel = relative_disparity_ratio(baseline_scores, scores)

        summary[lang_code] = {
            "mean": mean_val,
            "std": std_val,
            "absolute_gap": gap,
            "relative_disparity": rel,
        }

    return summary
