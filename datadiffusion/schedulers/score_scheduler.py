"""Score-based noise scheduler for 2-d tabular data."""

import torch
import numpy as np

from .noise_scheduler import NoiseScheduler
from ..tracking.logger import get_logger

logger = get_logger("datadiffusion.schedulers")


class ScoreBasedNoiseScheduler(NoiseScheduler):
    """Score-based SDE noise scheduler.

    Same schedule as ``NoiseScheduler``; ``step`` takes an Euler-Maruyama step
    using a predicted score instead of predicted noise.
    """

    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02, beta_schedule="linear"):
        super().__init__(num_timesteps, beta_start, beta_end, beta_schedule)
        self.sqrt_alphas_cumprod = torch.sqrt(torch.cumprod(1 - self.betas, dim=0))
        self.sqrt_1m_alphas_cumprod = torch.sqrt(1 - torch.cumprod(1 - self.betas, dim=0))

    def add_noise(self, x_start, noise, timesteps):
        """Forward process for 2-d tabular data: ``x_start``, ``noise`` of shape ``(B, F)``, ``timesteps`` of shape ``(B,)``."""
        sqrt_alphas_cumprod = self.sqrt_alphas_cumprod[timesteps].view(-1, 1)
        sqrt_one_minus = self.sqrt_1m_alphas_cumprod[timesteps].view(-1, 1)
        return sqrt_alphas_cumprod * x_start + sqrt_one_minus * noise

    def step(self, score, t, x):
        """One reverse SDE step from ``x`` (``(B, F)``) at integer step ``t`` given the predicted ``score``; returns ``(B, F)``."""
        dt = -1 / self.num_timesteps
        z = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
        drift = -0.5 * self.betas[t] * x - self.betas[t] * score
        diffusion = torch.sqrt(self.betas[t].clamp(min=1e-5))
        return x + drift * dt + diffusion * np.sqrt(abs(dt)) * z
