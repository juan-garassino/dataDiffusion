"""Tests for DDIM sampler."""

import torch
import numpy as np

from datadiffusion.schedulers import NoiseScheduler
from datadiffusion.models.factory import create_model_and_scheduler
from datadiffusion.generation.sampler import generate_samples_ddim
from datadiffusion.config import ExperimentConfig, ModelConfig, SchedulerConfig


def _make_model_and_scheduler():
    config = ExperimentConfig(
        model=ModelConfig(model_type="mlp", hidden_size=32, num_layers=1),
        scheduler=SchedulerConfig(num_timesteps=100),
    )
    model, scheduler = create_model_and_scheduler(config, input_size=4)
    return model, scheduler


def test_ddim_shape():
    model, scheduler = _make_model_and_scheduler()
    result = generate_samples_ddim(model, scheduler, num_samples=16, input_size=4, num_inference_steps=10)
    assert result.shape == (16, 4)


def test_ddim_deterministic():
    """DDIM with eta=0 should produce identical output with the same seed."""
    model, scheduler = _make_model_and_scheduler()

    torch.manual_seed(42)
    out1 = generate_samples_ddim(model, scheduler, num_samples=8, input_size=4, num_inference_steps=10, eta=0.0)

    torch.manual_seed(42)
    out2 = generate_samples_ddim(model, scheduler, num_samples=8, input_size=4, num_inference_steps=10, eta=0.0)

    np.testing.assert_allclose(out1, out2, atol=1e-5)
