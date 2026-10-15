"""Composite quality score and diagnostics."""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch

from .metrics import compute_feature_metrics, compute_correlation_diff
from .ml_utility import compute_ml_utility
from ..tracking.logger import get_logger

logger = get_logger("datadiffusion.evaluation")


@dataclass
class FeatureReport:
    """Per-feature fidelity statistics and a PASS/WARN/FAIL status.

    Status is derived from ``ks_statistic``: PASS below 0.15, WARN below 0.30,
    otherwise FAIL.
    """

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
        """Build a report from one dict returned by ``compute_feature_metrics``."""
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
    """Aggregate quality of synthetic data relative to real data.

    ``composite_score`` is in [0, 1]; higher is better. ``r2_real`` and
    ``r2_synthetic`` are 0.0 when no target was supplied.
    """

    feature_reports: List[FeatureReport]
    avg_ks_statistic: float
    corr_diff_norm: float
    ml_utility_ratio: float
    r2_real: float
    r2_synthetic: float
    composite_score: float

    @property
    def passed_features(self) -> List[FeatureReport]:
        """Feature reports with status ``PASS``."""
        return [f for f in self.feature_reports if f.status == "PASS"]

    @property
    def failed_features(self) -> List[FeatureReport]:
        """Feature reports with status ``FAIL``."""
        return [f for f in self.feature_reports if f.status == "FAIL"]


def compute_quality_metrics(
    real_data: np.ndarray,
    synthetic_data: np.ndarray,
    real_y: Optional[np.ndarray] = None,
    feature_names: Optional[List[str]] = None,
) -> QualityMetrics:
    """Compute the full quality assessment of synthetic vs real data.

    Parameters
    ----------
    real_data : numpy.ndarray or torch.Tensor
        Real features, shape ``(n_real, n_features)``.
    synthetic_data : numpy.ndarray or torch.Tensor
        Synthetic features, shape ``(n_synth, n_features)``.
    real_y : numpy.ndarray or torch.Tensor, optional
        Real target, shape ``(n_real, 1)`` or ``(n_real,)``. If omitted the ML
        utility term is 0.
    feature_names : list of str, optional
        Names used in the per-feature reports.

    Returns
    -------
    QualityMetrics
        ``composite_score = 0.3 * ks_score + 0.3 * corr_score + 0.4 * ml_score``.
    """

    if isinstance(real_data, torch.Tensor):
        real_data = real_data.detach().cpu().numpy()
    if isinstance(synthetic_data, torch.Tensor):
        synthetic_data = synthetic_data.detach().cpu().numpy()
    if isinstance(real_y, torch.Tensor):
        real_y = real_y.detach().cpu().numpy()

    raw_metrics = compute_feature_metrics(real_data, synthetic_data, feature_names)
    feature_reports = [FeatureReport.from_dict(m) for m in raw_metrics]

    avg_ks = np.mean([f.ks_statistic for f in feature_reports])

    corr_diff = compute_correlation_diff(real_data, synthetic_data)

    if real_y is not None:
        r2_real, r2_synth, utility_ratio = compute_ml_utility(
            real_data, real_y, synthetic_data
        )
    else:
        r2_real, r2_synth, utility_ratio = 0.0, 0.0, 0.0

    # Each component is scaled to [0, 1] with higher = better.
    ks_score = max(0.0, 1.0 - avg_ks)
    # sqrt(2) * n_features is a normalising constant, not a strict maximum of the Frobenius norm; the score is clamped at 0.
    n_features = real_data.shape[1]
    max_corr_norm = np.sqrt(2) * n_features
    corr_score = max(0.0, 1.0 - corr_diff / max_corr_norm)
    # The utility ratio can exceed 1 (synthetic beats real), so cap it.
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
