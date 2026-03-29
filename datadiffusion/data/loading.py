"""Dataset loading functions."""

import numpy as np
import torch
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from torch.utils.data import TensorDataset

from ..tracking.logger import get_logger

logger = get_logger("datadiffusion.data")

CALIFORNIA_FEATURE_NAMES = [
    "MedInc", "HouseAge", "AveRooms", "AveBedrms",
    "Population", "AveOccup", "Latitude", "Longitude",
]


def california_housing_dataset(scaler_type: str = "quantile"):
    """Load California Housing and return (TensorDataset, transformer).

    The transformer can be used to inverse-transform generated samples
    back to the original feature space.
    """
    logger.info("Loading California Housing dataset")
    california = fetch_california_housing()
    X, y = california.data, california.target

    if scaler_type == "quantile":
        transformer = QuantileTransformer(
            output_distribution="normal", n_quantiles=1000, random_state=0
        )
    else:
        transformer = StandardScaler()

    X_scaled = transformer.fit_transform(X)

    X_tensor = torch.from_numpy(X_scaled.astype(np.float32))
    y_tensor = torch.from_numpy(y.astype(np.float32)).reshape(-1, 1)

    logger.info("California Housing loaded. Shape: %s", X_tensor.shape)
    return TensorDataset(X_tensor, y_tensor), transformer


def get_dataset(name: str, scaler_type: str = "quantile"):
    """Return (TensorDataset, transformer) for the given dataset name."""
    if name == "california":
        return california_housing_dataset(scaler_type=scaler_type)
    raise ValueError(f"Unknown dataset: {name}")
