"""Statistical fidelity metrics: KS, Wasserstein, mean/std diff, correlation."""

import numpy as np
from scipy.stats import ks_2samp, wasserstein_distance

from ..tracking.logger import get_logger

logger = get_logger("datadiffusion.evaluation")


def compute_feature_metrics(real: np.ndarray, synthetic: np.ndarray, feature_names=None):
    """Per-feature statistical comparison.

    Returns list of dicts, one per feature, with KS, Wasserstein, mean/std diff.
    """
    n_features = real.shape[1]
    if feature_names is None:
        feature_names = [f"Feature_{i}" for i in range(n_features)]

    results = []
    for i in range(n_features):
        r, s = real[:, i], synthetic[:, i]
        ks_stat, ks_p = ks_2samp(r, s)
        w_dist = wasserstein_distance(r, s)
        results.append({
            "feature_index": i,
            "feature_name": feature_names[i],
            "ks_statistic": float(ks_stat),
            "ks_pvalue": float(ks_p),
            "wasserstein_distance": float(w_dist),
            "mean_diff": float(abs(r.mean() - s.mean())),
            "std_diff": float(abs(r.std() - s.std())),
        })
    return results


def compute_correlation_diff(real: np.ndarray, synthetic: np.ndarray) -> float:
    """Frobenius norm of the difference between correlation matrices."""
    corr_real = np.corrcoef(real.T)
    corr_synth = np.corrcoef(synthetic.T)
    return float(np.linalg.norm(corr_real - corr_synth, "fro"))
