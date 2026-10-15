"""Data normalization utilities."""

import numpy as np
import torch


def normalize_data(data):
    """Standardise columns to zero mean and unit variance.

    Parameters
    ----------
    data : numpy.ndarray or torch.Tensor
        Array of shape ``(n_samples, n_features)``.

    Returns
    -------
    normalized, mean, std
        ``normalized`` has the shape and type of ``data``; ``mean`` and ``std``
        have shape ``(n_features,)``. ``numpy`` input uses the population std
        (``ddof=0``), ``torch`` input the unbiased std.

    Raises
    ------
    TypeError
        If ``data`` is neither an ndarray nor a Tensor.
    """
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
    """Invert ``normalize_data``: ``normalized_data * std + mean``."""
    return normalized_data * std + mean
