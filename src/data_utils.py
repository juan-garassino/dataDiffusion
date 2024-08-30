# data_utils.py

import torch
import numpy as np
import pandas as pd
from .custom_logger import get_default_logger

logger = get_default_logger(verbose=False)

def prepare_data(X, y=None, verbose=False):
    logger.info("Preparing data")

    if isinstance(X, pd.DataFrame):
        X_tensor = torch.tensor(X.values, dtype=torch.float32)
    else:
        X_tensor = torch.tensor(X, dtype=torch.float32)

    if y is not None:
        if isinstance(y, pd.Series):
            y_tensor = torch.tensor(y.values, dtype=torch.float32).reshape(-1, 1)
        else:
            y_tensor = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
        
        logger.info(f"Prepared data shapes - X: {X_tensor.shape}, y: {y_tensor.shape}")
        return X_tensor, y_tensor
    else:
        logger.info(f"Prepared data shape - X: {X_tensor.shape}")
        return X_tensor

def normalize_data(data):
    logger.info("Normalizing data")
    if isinstance(data, pd.DataFrame):
        mean = data.mean()
        std = data.std()
        normalized_data = (data - mean) / std
    elif isinstance(data, np.ndarray):
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        normalized_data = (data - mean) / std
    elif isinstance(data, torch.Tensor):
        mean = torch.mean(data, dim=0)
        std = torch.std(data, dim=0)
        normalized_data = (data - mean) / std
    else:
        raise TypeError("Unsupported data type for normalization")
    
    logger.info(f"Data normalized. Shape: {normalized_data.shape}")
    return normalized_data, mean, std

def denormalize_data(normalized_data, mean, std):
    logger.info("Denormalizing data")
    denormalized_data = normalized_data * std + mean
    logger.info(f"Data denormalized. Shape: {denormalized_data.shape}")
    return denormalized_data

def safe_normalize(x, dim=1, eps=1e-8):
    mean = x.mean(dim=dim, keepdim=True)
    var = x.var(dim=dim, keepdim=True, unbiased=False)
    std = torch.sqrt(var + eps)

    logger.debug(f"Safe normalization stats:")
    logger.debug(f"  Mean - min: {mean.min().item():.4f}, max: {mean.max().item():.4f}")
    logger.debug(f"  Var  - min: {var.min().item():.4f}, max: {var.max().item():.4f}")
    logger.debug(f"  Std  - min: {std.min().item():.4f}, max: {std.max().item():.4f}")

    # Check for any problematic values
    if torch.isnan(std).any() or torch.isinf(std).any() or (std == 0).any():
        logger.warning("Detected NaN, Inf, or zero values in std. Using alternative normalization.")
        return x - mean

    return (x - mean) / std