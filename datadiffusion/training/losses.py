"""Loss functions for diffusion training.

BUG FIX #2: train.py hardcoded score loss for all model types.
BUG FIX #3: train.py referenced scheduler.sqrt_1m_alphas_cumprod — only on ScoreBasedNoiseScheduler.
Now: get_loss_fn() selects the correct loss based on model_type.
"""

import torch
import torch.nn.functional as F


def noise_prediction_loss(predicted: torch.Tensor, noise: torch.Tensor, **_kwargs) -> torch.Tensor:
    """Standard DDPM noise prediction loss (for MLP / Transformer)."""
    return F.mse_loss(predicted, noise)


def score_matching_loss(
    predicted: torch.Tensor,
    noise: torch.Tensor,
    sqrt_1m_alpha: torch.Tensor,
    timesteps: torch.Tensor,
    **_kwargs,
) -> torch.Tensor:
    """Score matching loss — computes target score from noise."""
    target = -noise / (sqrt_1m_alpha[timesteps].view(-1, 1) + 1e-8)
    return F.mse_loss(predicted, target)


def get_loss_fn(model_type: str):
    """Return the correct loss function for the model type."""
    if model_type in ("mlp", "transformer"):
        return noise_prediction_loss
    elif model_type == "score":
        return score_matching_loss
    else:
        raise ValueError(f"Unknown model type for loss selection: {model_type}")
