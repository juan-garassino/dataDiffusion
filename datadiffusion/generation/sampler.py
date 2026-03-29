"""Unified sample generation from a trained diffusion model."""

import numpy as np
import torch
from tqdm import tqdm

from ..tracking.logger import get_logger

logger = get_logger("datadiffusion.generation")


@torch.no_grad()
def generate_samples(model, scheduler, num_samples: int, input_size: int, num_timesteps: int = None):
    """Generate synthetic samples via the reverse diffusion process (DDPM).

    Returns numpy array of shape (num_samples, input_size).
    """
    if num_timesteps is None:
        num_timesteps = scheduler.num_timesteps

    device = next(model.parameters()).device
    model.eval()

    sample = torch.randn(num_samples, input_size, device=device)

    for t in tqdm(range(num_timesteps - 1, -1, -1), desc="Generating", leave=False):
        t_batch = torch.full((num_samples,), t, device=device, dtype=torch.long)
        predicted = model(sample, t_batch.float())
        sample = scheduler.step(predicted, t, sample)

    logger.info("Generated %d samples of dim %d", num_samples, input_size)
    return sample.cpu().numpy()


@torch.no_grad()
def generate_samples_ddim(
    model, scheduler, num_samples: int, input_size: int,
    num_inference_steps: int = 50, eta: float = 0.0,
):
    """Generate synthetic samples via DDIM (deterministic when eta=0).

    Returns numpy array of shape (num_samples, input_size).
    """
    device = next(model.parameters()).device
    model.eval()

    # Build a sub-sequence of timesteps evenly spaced over [0, T)
    total_timesteps = scheduler.num_timesteps
    step_ratio = total_timesteps // num_inference_steps
    timesteps = (np.arange(0, num_inference_steps) * step_ratio).astype(np.int64)
    timesteps = np.flip(timesteps).copy()  # reverse: high → low

    alphas_cumprod = scheduler.alphas_cumprod.to(device)

    sample = torch.randn(num_samples, input_size, device=device)

    for i, t in enumerate(tqdm(timesteps, desc="DDIM Generating", leave=False)):
        t_batch = torch.full((num_samples,), t, device=device, dtype=torch.long)
        predicted_noise = model(sample, t_batch.float())

        alpha_t = alphas_cumprod[t]
        alpha_prev = alphas_cumprod[timesteps[i + 1]] if i + 1 < len(timesteps) else torch.tensor(1.0, device=device)

        # DDIM update rule
        pred_x0 = (sample - (1 - alpha_t).sqrt() * predicted_noise) / alpha_t.sqrt()

        sigma_t = eta * ((1 - alpha_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_prev)).sqrt()
        dir_xt = (1 - alpha_prev - sigma_t ** 2).sqrt() * predicted_noise
        sample = alpha_prev.sqrt() * pred_x0 + dir_xt

        if sigma_t > 0:
            sample = sample + sigma_t * torch.randn_like(sample)

    logger.info("DDIM generated %d samples of dim %d (%d steps, eta=%.1f)",
                num_samples, input_size, num_inference_steps, eta)
    return sample.cpu().numpy()
