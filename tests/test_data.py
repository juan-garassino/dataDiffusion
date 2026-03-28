"""Tests for data loading and normalization."""

import numpy as np
import torch

from datadiffusion.data.loading import california_housing_dataset
from datadiffusion.data.utils import normalize_data, denormalize_data


def test_california_loads():
    ds, transformer = california_housing_dataset()
    X = ds.tensors[0]
    assert X.shape == (20640, 8), f"Expected (20640, 8) but got {X.shape}"
    assert ds.tensors[1].shape == (20640, 1)


def test_california_standard_scaler():
    ds, transformer = california_housing_dataset(scaler_type="standard")
    X = ds.tensors[0]
    assert X.shape == (20640, 8)


def test_quantile_roundtrip():
    """QuantileTransformer should faithfully recover the vast majority of data."""
    ds, transformer = california_housing_dataset(scaler_type="quantile")
    X_scaled = ds.tensors[0].numpy()
    X_recovered = transformer.inverse_transform(X_scaled)
    # Re-load raw data for comparison
    from sklearn.datasets import fetch_california_housing
    X_raw = fetch_california_housing().data
    # QuantileTransformer clips extreme outliers, so check that >99.9% of
    # values round-trip within tolerance
    close = np.abs(X_recovered - X_raw) < 0.5
    pct_close = close.mean()
    assert pct_close > 0.999, f"Only {pct_close:.4%} of values round-tripped within tolerance"


def test_normalize_denormalize_roundtrip():
    data = np.random.randn(100, 5).astype(np.float32)
    normed, mean, std = normalize_data(data)
    recovered = denormalize_data(normed, mean, std)
    np.testing.assert_allclose(data, recovered, atol=1e-5)


def test_normalize_torch():
    data = torch.randn(100, 5)
    normed, mean, std = normalize_data(data)
    assert normed.shape == data.shape
    recovered = denormalize_data(normed, mean, std)
    torch.testing.assert_close(data, recovered, atol=1e-5, rtol=1e-5)
