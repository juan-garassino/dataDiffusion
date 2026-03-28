"""Tests for noise schedulers — includes 2D shape regression test for BUG #1."""

import pytest
import torch

from datadiffusion.schedulers import NoiseScheduler, ScoreBasedNoiseScheduler


@pytest.fixture
def scheduler():
    return NoiseScheduler(num_timesteps=1000, beta_start=0.0001, beta_end=0.02)


@pytest.fixture
def score_scheduler():
    return ScoreBasedNoiseScheduler(num_timesteps=1000, beta_start=0.0001, beta_end=0.02)


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
    x_prev = scheduler.step(model_output, 500, x_t)
    assert x_prev.shape == x_t.shape


def test_score_scheduler_2d_tabular(score_scheduler):
    """Regression test for BUG #1: add_noise must work with 2D tabular data."""
    x_start = torch.randn(32, 8)
    noise = torch.randn_like(x_start)
    timesteps = torch.randint(0, 1000, (32,))
    x_noisy = score_scheduler.add_noise(x_start, noise, timesteps)
    assert x_noisy.shape == (32, 8), f"Expected (32, 8) but got {x_noisy.shape}"


def test_score_scheduler_step(score_scheduler):
    x = torch.randn(10, 8)
    score = torch.randn_like(x)
    x_prev = score_scheduler.step(score, 500, x)
    assert x_prev.shape == x.shape


def test_len(scheduler):
    assert len(scheduler) == 1000


def test_cosine_schedule_monotonic():
    """Cosine schedule should produce monotonically decreasing alphas_cumprod."""
    sched = NoiseScheduler(num_timesteps=1000, beta_schedule="cosine")
    ac = sched.alphas_cumprod
    assert ac.shape == (1000,)
    # alphas_cumprod should be strictly decreasing
    diffs = ac[1:] - ac[:-1]
    assert (diffs < 0).all(), "alphas_cumprod should be monotonically decreasing"
    # Should start near 1 and end near 0
    assert ac[0] > 0.99
    assert ac[-1] < 0.05
