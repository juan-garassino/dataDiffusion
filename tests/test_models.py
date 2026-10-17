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


@pytest.mark.parametrize("model_type,scheduler_type,warns", [
    ("score", "standard", True),
    ("transformer", "score", True),
    ("mlp", "score", True),
    ("score", "score", False),
    ("mlp", "standard", False),
])
def test_factory_warns_on_mismatched_model_scheduler(caplog, model_type, scheduler_type, warns):
    """Score models output a score, the other models output noise; the sampler step must match (G-010)."""
    import logging

    from datadiffusion.config import ExperimentConfig, ModelConfig, SchedulerConfig
    from datadiffusion.models.factory import create_model_and_scheduler

    config = ExperimentConfig(
        model=ModelConfig(model_type=model_type, hidden_size=32, num_layers=1),
        scheduler=SchedulerConfig(scheduler_type=scheduler_type, num_timesteps=10),
    )
    with caplog.at_level(logging.WARNING, logger="datadiffusion"):
        create_model_and_scheduler(config, input_size=4)
    assert any("mismatch" in r.getMessage() for r in caplog.records) == warns
