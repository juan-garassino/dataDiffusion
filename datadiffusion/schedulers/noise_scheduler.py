"""Standard DDPM noise scheduler — no side effects in constructor."""

import torch
import torch.nn.functional as F
import numpy as np

from ..tracking.logger import get_logger

logger = get_logger("datadiffusion.schedulers")


class NoiseScheduler:
    """DDPM noise scheduler.

    Precomputes the beta schedule and derived terms as 1-d float32 tensors of
    length ``num_timesteps``.

    Parameters
    ----------
    num_timesteps : int
        Number of diffusion steps ``T``.
    beta_start, beta_end : float
        Endpoints for the ``linear`` and ``quadratic`` schedules (unused by ``cosine``).
    beta_schedule : {"linear", "quadratic", "cosine"}
        Noise schedule.

    Raises
    ------
    ValueError
        For an unknown ``beta_schedule``.
    """

    def __init__(
        self,
        num_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="linear",
    ):
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_schedule = beta_schedule

        if beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float32)
        elif beta_schedule == "quadratic":
            self.betas = (
                torch.linspace(beta_start**0.5, beta_end**0.5, num_timesteps, dtype=torch.float32) ** 2
            )
        elif beta_schedule == "cosine":
            # Nichol & Dhariwal (Improved DDPM) cosine schedule
            s = 0.008
            steps = num_timesteps + 1
            t = torch.linspace(0, num_timesteps, steps, dtype=torch.float64)
            alphas_cumprod = torch.cos(((t / num_timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            self.betas = torch.clamp(betas, min=0.0, max=0.999).float()
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        self.sqrt_alphas_cumprod = self.alphas_cumprod**0.5
        self.sqrt_one_minus_alphas_cumprod = (1 - self.alphas_cumprod) ** 0.5

        self.sqrt_inv_alphas_cumprod = torch.sqrt(1 / self.alphas_cumprod)
        self.sqrt_inv_alphas_cumprod_minus_one = torch.sqrt(1 / self.alphas_cumprod - 1)

        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        )

        logger.info("NoiseScheduler initialised: %d timesteps", num_timesteps)

    def reconstruct_x0(self, x_t, t, noise):
        """Estimate the clean sample from ``x_t`` of shape ``(B, F)``, integer timesteps ``t`` of shape ``(B,)`` and predicted ``noise`` of shape ``(B, F)``."""
        s1 = self.sqrt_inv_alphas_cumprod[t].view(-1, 1)
        s2 = self.sqrt_inv_alphas_cumprod_minus_one[t].view(-1, 1)
        return s1 * x_t - s2 * noise

    def q_posterior(self, x_0, x_t, t):
        """Mean of ``q(x_{t-1} | x_t, x_0)``; shapes as in ``reconstruct_x0``."""
        s1 = self.posterior_mean_coef1[t].view(-1, 1)
        s2 = self.posterior_mean_coef2[t].view(-1, 1)
        return s1 * x_0 + s2 * x_t

    def get_variance(self, t):
        """Posterior variance at integer timestep ``t`` (0 at ``t == 0``, otherwise clamped to >= 1e-20)."""
        if t == 0:
            return 0
        variance = (
            self.betas[t] * (1.0 - self.alphas_cumprod_prev[t]) / (1.0 - self.alphas_cumprod[t])
        )
        return variance.clamp(min=1e-20)

    def step(self, model_output, timestep, sample):
        """One DDPM reverse step.

        Parameters
        ----------
        model_output : torch.Tensor
            Predicted noise, shape ``(B, F)``.
        timestep : int
            Current step ``t`` (a single value for the whole batch).
        sample : torch.Tensor
            Current ``x_t``, shape ``(B, F)``.

        Returns
        -------
        torch.Tensor
            ``x_{t-1}``, shape ``(B, F)``.
        """
        t = timestep
        pred_original_sample = self.reconstruct_x0(sample, t, model_output)
        pred_prev_sample = self.q_posterior(pred_original_sample, sample, t)

        variance = 0
        if t > 0:
            noise = torch.randn_like(model_output)
            variance = (self.get_variance(t) ** 0.5) * noise

        return pred_prev_sample + variance

    def add_noise(self, x_start, x_noise, timesteps):
        """Forward process: ``sqrt(a_bar_t) * x_start + sqrt(1 - a_bar_t) * x_noise``.

        ``x_start`` and ``x_noise`` have shape ``(B, F)``; ``timesteps`` is a
        long tensor of shape ``(B,)``.
        """
        s1 = self.sqrt_alphas_cumprod[timesteps].view(-1, 1)
        s2 = self.sqrt_one_minus_alphas_cumprod[timesteps].view(-1, 1)
        return s1 * x_start + s2 * x_noise

    def __len__(self):
        """Number of diffusion timesteps."""
        return self.num_timesteps
