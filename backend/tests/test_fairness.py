"""Unit tests for Cross-Lingual Fairness and Disparity metrics."""

import pytest
from app.core.fairness import (
    absolute_performance_gap,
    relative_disparity_ratio,
    transliteration_penalty,
    compute_disparity_summary,
)


def test_absolute_performance_gap():
    baseline_en = [0.8, 0.9, 0.7]  # mean = 0.80
    target_bn = [0.5, 0.6, 0.4]    # mean = 0.50

    # Delta_abs = 0.80 - 0.50 = 0.30
    gap = absolute_performance_gap(baseline_en, target_bn)
    assert gap == pytest.approx(0.30)

    # Exact parity
    assert absolute_performance_gap(baseline_en, baseline_en) == 0.0

    # Target outperforming baseline
    superior_target = [0.9, 1.0, 0.8]  # mean = 0.90
    assert absolute_performance_gap(baseline_en, superior_target) == pytest.approx(-0.10)


def test_relative_disparity_ratio():
    baseline_en = [0.8, 0.9, 0.7]  # mean = 0.80
    target_bn = [0.4, 0.4, 0.4]    # mean = 0.40

    # D_rel = 0.40 / 0.80 = 0.50
    ratio = relative_disparity_ratio(baseline_en, target_bn)
    assert ratio == pytest.approx(0.50)

    # Perfect parity gives 1.0
    assert relative_disparity_ratio(baseline_en, baseline_en) == pytest.approx(1.0)


def test_transliteration_penalty():
    # Native Bengali script (bn) vs Latin transliteration (bn-Latn)
    native_bn = [0.85, 0.75]        # mean = 0.80
    translit_bn = [0.35, 0.25]      # mean = 0.30

    # Drop attributable to transliteration: 0.80 - 0.30 = 0.50
    penalty = transliteration_penalty(native_bn, translit_bn)
    assert penalty == pytest.approx(0.50)


def test_compute_disparity_summary():
    data = {
        "en": [0.9, 0.8],        # mean = 0.85
        "bn": [0.7, 0.6],        # mean = 0.65, gap = 0.20
        "bn-Latn": [0.4, 0.3],   # mean = 0.35, gap = 0.50
    }

    summary = compute_disparity_summary(data, baseline_language="en")

    assert "en" in summary
    assert "bn" in summary
    assert "bn-Latn" in summary

    assert summary["en"]["absolute_gap"] == 0.0
    assert summary["en"]["relative_disparity"] == pytest.approx(1.0)

    assert summary["bn"]["mean"] == pytest.approx(0.65)
    assert summary["bn"]["absolute_gap"] == pytest.approx(0.20)
    assert summary["bn"]["relative_disparity"] == pytest.approx(0.65 / 0.85)

    assert summary["bn-Latn"]["mean"] == pytest.approx(0.35)
    assert summary["bn-Latn"]["absolute_gap"] == pytest.approx(0.50)


def test_fairness_validation_and_errors():
    with pytest.raises(ValueError, match="mismatch"):
        absolute_performance_gap([0.8, 0.9], [0.5])

    with pytest.raises(ValueError, match="not found"):
        compute_disparity_summary({"bn": [0.5]}, baseline_language="en")
