import pytest
import torch
import numpy as np
from src.scheduler import NoiseScheduler


@pytest.fixture
def scheduler():
    return NoiseScheduler(num_timesteps=1000, beta_start=0.0001, beta_end=0.02)


def test_scheduler_initialization(scheduler):
    assert scheduler.num_timesteps == 1000
    assert len(scheduler.betas) == 1000


def test_add_noise(scheduler):
    x_start = torch.randn(10, 5)
    x_noise = torch.randn_like(x_start)
    timesteps = torch.randint(0, len(scheduler), (10,))
    x_t = scheduler.add_noise(x_start, x_noise, timesteps)
    assert x_t.shape == x_start.shape


def test_step(scheduler):
    x_t = torch.randn(10, 5)
    model_output = torch.randn_like(x_t)
    timestep = 500
    x_t_minus_1 = scheduler.step(model_output, timestep, x_t)
    assert x_t_minus_1.shape == x_t.shape


def test_normalize_data(scheduler):
    data = np.random.randn(100, 5)
    normalized_data, mean, std = scheduler.normalize_data(data)
    assert normalized_data.shape == data.shape
    assert np.isclose(normalized_data.mean(), 0, atol=1e-6)
    assert np.isclose(normalized_data.std(), 1, atol=1e-6)
