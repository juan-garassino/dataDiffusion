# data_utils.py
import torch
import numpy as np
import pandas as pd
from .custom_logger import get_default_logger


def prepare_data(X, y, verbose=False):
    logger = get_default_logger(verbose)
    logger.info("Preparing data")
    X_tensor = torch.tensor(
        X.values if hasattr(X, "values") else X, dtype=torch.float32
    )
    y_tensor = torch.tensor(
        y.values if hasattr(y, "values") else y, dtype=torch.float32
    ).reshape(-1, 1)
    logger.info(f"Prepared data shapes - X: {X_tensor.shape}, y: {y_tensor.shape}")
    return X_tensor, y_tensor


def normalize_data(data):
    if isinstance(data, pd.DataFrame):
        mean = data.mean()
        std = data.std()
        normalized_data = (data - mean) / std
    else:
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        normalized_data = (data - mean) / std
    return normalized_data, mean, std


def denormalize_data(normalized_data, mean, std):
    return normalized_data * std + mean
