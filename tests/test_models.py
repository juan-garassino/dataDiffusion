"""Tests for model forward passes and output shapes."""

import pytest
import torch

from datadiffusion.models.mlp import TabularMLP
from datadiffusion.models.transformer import TabularTransformer
from datadiffusion.models.score import ScoreNetwork, EnhancedScoreNetwork


@pytest.mark.parametrize("embedding_type", ["sinusoidal", "linear", "learnable", "identity", "zero"])
def test_mlp_forward(embedding_type):
    model = TabularMLP(input_size=8, hidden_size=32, hidden_layers=2, embedding_type=embedding_type)
    x = torch.randn(16, 8)
    t = torch.randint(0, 100, (16,))
    out = model(x, t)
    assert out.shape == (16, 8)
    assert not torch.isnan(out).any()


def test_transformer_forward():
    model = TabularTransformer(input_size=8, hidden_size=32, num_layers=1, num_heads=2)
    x = torch.randn(16, 8)
    t = torch.randint(0, 100, (16,))
    out = model(x, t)
    assert out.shape == (16, 8)
    assert not torch.isnan(out).any()


def test_score_network_forward():
    model = ScoreNetwork(input_dim=8, hidden_dim=32, num_layers=2)
    x = torch.randn(16, 8)
    t = torch.randint(0, 100, (16,))
    out = model(x, t)
    assert out.shape == (16, 8)


def test_enhanced_score_network_forward():
    model = EnhancedScoreNetwork(input_dim=8, hidden_dim=32, num_layers=2, num_heads=2)
    x = torch.randn(16, 8)
    t = torch.randint(0, 100, (16,))
    out = model(x, t)
    assert out.shape == (16, 8)
