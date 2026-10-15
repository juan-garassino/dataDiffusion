"""ML utility test: train-on-synthetic, test-on-real."""

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from ..tracking.logger import get_logger

logger = get_logger("datadiffusion.evaluation")


def compute_ml_utility(real_X: np.ndarray, real_y: np.ndarray, synthetic_X: np.ndarray, synthetic_y: np.ndarray = None):
    """Compute ML utility ratio: R2_synthetic / R2_real.

    Both models are ``GradientBoostingRegressor`` instances evaluated on the
    same 20% held-out slice of the real data.

    Parameters
    ----------
    real_X : numpy.ndarray
        Real features, shape ``(n_real, n_features)``.
    real_y : numpy.ndarray
        Real target, shape ``(n_real, 1)`` or ``(n_real,)``.
    synthetic_X : numpy.ndarray
        Synthetic features, shape ``(n_synth, n_features)``.
    synthetic_y : numpy.ndarray, optional
        Synthetic target, shape ``(n_synth,)`` or ``(n_synth, 1)``. If omitted,
        the synthetic target is the prediction of the model trained on real data.

    Returns
    -------
    tuple of float
        ``(r2_real, r2_synthetic, utility_ratio)``. The ratio is 0.0 when
        ``r2_real <= 0`` and is clamped at 0 from below (it is not capped at 1).
    """
    X_train, X_test, y_train, y_test = train_test_split(real_X, real_y, test_size=0.2, random_state=42)

    model_real = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)
    model_real.fit(X_train, y_train.ravel())
    r2_real = r2_score(y_test, model_real.predict(X_test))

    if synthetic_y is None:
        # No real label exists for synthetic rows; label them with the real-data model.
        synthetic_y = model_real.predict(synthetic_X)

    model_synth = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)
    model_synth.fit(synthetic_X, synthetic_y.ravel())
    r2_synthetic = r2_score(y_test, model_synth.predict(X_test))

    # A non-positive R2_real makes the ratio meaningless, so report 0.
    if r2_real <= 0:
        utility_ratio = 0.0
    else:
        utility_ratio = max(0.0, r2_synthetic / r2_real)

    logger.info("ML utility — R2_real=%.4f  R2_synth=%.4f  ratio=%.4f", r2_real, r2_synthetic, utility_ratio)
    return r2_real, r2_synthetic, utility_ratio
