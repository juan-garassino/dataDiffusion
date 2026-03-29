"""Score-based noise scheduler — fixes 4D reshape bug (#1)."""

import torch
import numpy as np

from .noise_scheduler import NoiseScheduler
from ..tracking.logger import get_logger

logger = get_logger("datadiffusion.schedulers")


class ScoreBasedNoiseScheduler(NoiseScheduler):
    """Score-based SDE noise scheduler.

    BUG FIX #1: add_noise used .view(-1,1,1,1) — wrong for 2D tabular data.
    Now uses .view(-1, 1).
    """

    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02, beta_schedule="linear"):
        super().__init__(num_timesteps, beta_start, beta_end, beta_schedule)
        self.sqrt_alphas_cumprod = torch.sqrt(torch.cumprod(1 - self.betas, dim=0))
        self.sqrt_1m_alphas_cumprod = torch.sqrt(1 - torch.cumprod(1 - self.betas, dim=0))

    def add_noise(self, x_start, noise, timesteps):
        # FIX: .view(-1, 1) for 2D tabular data (was .view(-1,1,1,1))
        sqrt_alphas_cumprod = self.sqrt_alphas_cumprod[timesteps].view(-1, 1)
        sqrt_one_minus = self.sqrt_1m_alphas_cumprod[timesteps].view(-1, 1)
        return sqrt_alphas_cumprod * x_start + sqrt_one_minus * noise

    def step(self, score, t, x):
        dt = -1 / self.num_timesteps
        z = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
        drift = -0.5 * self.betas[t] * x - self.betas[t] * score
        diffusion = torch.sqrt(self.betas[t].clamp(min=1e-5))
        return x + drift * dt + diffusion * np.sqrt(abs(dt)) * z
