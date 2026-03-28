"""Composite quality score and diagnostics."""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch

from .metrics import compute_feature_metrics, compute_correlation_diff
from .ml_utility import compute_ml_utility
from ..tracking.logger import get_logger

logger = get_logger("datadiffusion.evaluation")


@dataclass
class FeatureReport:
    feature_index: int
    feature_name: str
    ks_statistic: float
    ks_pvalue: float
    wasserstein_distance: float
    mean_diff: float
    std_diff: float
    status: str  # PASS / WARN / FAIL

    @classmethod
    def from_dict(cls, d: dict) -> "FeatureReport":
        ks = d["ks_statistic"]
        p = d["ks_pvalue"]
        if ks < 0.15:
            status = "PASS"
        elif ks < 0.30:
            status = "WARN"
        else:
            status = "FAIL"
        return cls(
            feature_index=d["feature_index"],
            feature_name=d["feature_name"],
            ks_statistic=ks,
            ks_pvalue=p,
            wasserstein_distance=d["wasserstein_distance"],
            mean_diff=d["mean_diff"],
            std_diff=d["std_diff"],
            status=status,
        )


@dataclass
class QualityMetrics:
    feature_reports: List[FeatureReport]
    avg_ks_statistic: float
    corr_diff_norm: float
    ml_utility_ratio: float
    r2_real: float
    r2_synthetic: float
    composite_score: float

    @property
    def passed_features(self):
        return [f for f in self.feature_reports if f.status == "PASS"]

    @property
    def failed_features(self):
        return [f for f in self.feature_reports if f.status == "FAIL"]


def compute_quality_metrics(
    real_data: np.ndarray,
    synthetic_data: np.ndarray,
    real_y: Optional[np.ndarray] = None,
    feature_names: Optional[list] = None,
) -> QualityMetrics:
    """Compute the full quality assessment of synthetic vs real data."""

    # Convert tensors
    if isinstance(real_data, torch.Tensor):
        real_data = real_data.detach().cpu().numpy()
    if isinstance(synthetic_data, torch.Tensor):
        synthetic_data = synthetic_data.detach().cpu().numpy()
    if isinstance(real_y, torch.Tensor):
        real_y = real_y.detach().cpu().numpy()

    # Per-feature metrics
    raw_metrics = compute_feature_metrics(real_data, synthetic_data, feature_names)
    feature_reports = [FeatureReport.from_dict(m) for m in raw_metrics]

    # Aggregate KS
    avg_ks = np.mean([f.ks_statistic for f in feature_reports])

    # Correlation structure
    corr_diff = compute_correlation_diff(real_data, synthetic_data)

    # ML utility
    if real_y is not None:
        r2_real, r2_synth, utility_ratio = compute_ml_utility(
            real_data, real_y, synthetic_data
        )
    else:
        r2_real, r2_synth, utility_ratio = 0.0, 0.0, 0.0

    # Composite score: weighted combination
    # KS component: 1 - avg_ks (lower KS = better), clamped to [0, 1]
    ks_score = max(0.0, 1.0 - avg_ks)
    # Correlation component: 1 - normalized corr_diff
    n_features = real_data.shape[1]
    max_corr_norm = np.sqrt(2) * n_features  # theoretical max
    corr_score = max(0.0, 1.0 - corr_diff / max_corr_norm)
    # ML utility component: already in [0, 1] (clamped)
    ml_score = min(1.0, utility_ratio)

    composite = 0.3 * ks_score + 0.3 * corr_score + 0.4 * ml_score

    logger.info(
        "Quality — KS=%.3f  corr_diff=%.3f  ml_ratio=%.3f  composite=%.3f",
        avg_ks, corr_diff, utility_ratio, composite,
    )

    return QualityMetrics(
        feature_reports=feature_reports,
        avg_ks_statistic=avg_ks,
        corr_diff_norm=corr_diff,
        ml_utility_ratio=utility_ratio,
        r2_real=r2_real,
        r2_synthetic=r2_synth,
        composite_score=composite,
    )
