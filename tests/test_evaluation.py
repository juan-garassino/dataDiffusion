"""Tests for evaluation pipeline — composite score, KS, ML utility."""

import numpy as np
import pytest

from datadiffusion.evaluation.metrics import compute_feature_metrics, compute_correlation_diff
from datadiffusion.evaluation.ml_utility import compute_ml_utility
from datadiffusion.evaluation.composite import compute_quality_metrics, QualityMetrics


@pytest.fixture
def real_and_synthetic():
    rng = np.random.RandomState(42)
    real = rng.randn(500, 8)
    # Synthetic is slightly shifted — should still be evaluable
    synthetic = rng.randn(500, 8) * 0.9 + 0.1
    y = rng.randn(500, 1)
    return real, synthetic, y


def test_feature_metrics(real_and_synthetic):
    real, synthetic, _ = real_and_synthetic
    results = compute_feature_metrics(real, synthetic)
    assert len(results) == 8
    for r in results:
        assert 0.0 <= r["ks_statistic"] <= 1.0
        assert 0.0 <= r["ks_pvalue"] <= 1.0


def test_correlation_diff(real_and_synthetic):
    real, synthetic, _ = real_and_synthetic
    diff = compute_correlation_diff(real, synthetic)
    assert diff >= 0.0


def test_ml_utility(real_and_synthetic):
    real, synthetic, y = real_and_synthetic
    r2_real, r2_synth, ratio = compute_ml_utility(real, y, synthetic)
    assert isinstance(ratio, float)


def test_composite_score(real_and_synthetic):
    real, synthetic, y = real_and_synthetic
    metrics = compute_quality_metrics(real, synthetic, y)
    assert isinstance(metrics, QualityMetrics)
    assert 0.0 <= metrics.composite_score <= 1.0
    assert len(metrics.feature_reports) == 8
