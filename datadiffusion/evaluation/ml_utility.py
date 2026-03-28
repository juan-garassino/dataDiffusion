"""ML utility test: train-on-synthetic, test-on-real."""

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from ..tracking.logger import get_logger

logger = get_logger("datadiffusion.evaluation")


def compute_ml_utility(real_X: np.ndarray, real_y: np.ndarray, synthetic_X: np.ndarray, synthetic_y: np.ndarray = None):
    """Compute ML utility ratio: R2_synthetic / R2_real.

    If synthetic_y is not provided, uses first column of synthetic_X as target
    (same convention as real_y).

    Returns (r2_real, r2_synthetic, utility_ratio).
    """
    # Split real data
    X_train, X_test, y_train, y_test = train_test_split(real_X, real_y, test_size=0.2, random_state=42)

    # Train on real
    model_real = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)
    model_real.fit(X_train, y_train.ravel())
    r2_real = r2_score(y_test, model_real.predict(X_test))

    # Train on synthetic
    if synthetic_y is None:
        # Generate a pseudo-target from synthetic features using the real model's predictions
        synthetic_y = model_real.predict(synthetic_X)

    model_synth = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)
    model_synth.fit(synthetic_X, synthetic_y.ravel())
    r2_synthetic = r2_score(y_test, model_synth.predict(X_test))

    # Avoid division by zero
    if r2_real <= 0:
        utility_ratio = 0.0
    else:
        utility_ratio = max(0.0, r2_synthetic / r2_real)

    logger.info("ML utility — R2_real=%.4f  R2_synth=%.4f  ratio=%.4f", r2_real, r2_synthetic, utility_ratio)
    return r2_real, r2_synthetic, utility_ratio
