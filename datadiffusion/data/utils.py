"""Data normalization utilities."""

import numpy as np
import torch


def normalize_data(data):
    if isinstance(data, np.ndarray):
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        return (data - mean) / (std + 1e-8), mean, std
    if isinstance(data, torch.Tensor):
        mean = torch.mean(data, dim=0)
        std = torch.std(data, dim=0)
        return (data - mean) / (std + 1e-8), mean, std
    raise TypeError(f"Unsupported data type: {type(data)}")


def denormalize_data(normalized_data, mean, std):
    return normalized_data * std + mean
