"""Positional / time embeddings for diffusion models."""

import torch
from torch import nn


class SinusoidalEmbedding(nn.Module):
    """Transformer-style sin/cos embedding of a scalar timestep.

    Parameters
    ----------
    size : int
        Output dimension (even; ``size // 2`` frequencies for sin and for cos).
    scale : float
        Multiplier applied to the input before embedding.
    """

    def __init__(self, size: int, scale: float = 1.0):
        super().__init__()
        self.size = size
        self.scale = scale

    def forward(self, x: torch.Tensor):
        """Embed timesteps ``x`` of shape ``(B,)`` or ``(B, 1)``; returns ``(B, size)`` or ``(B, 1, size)``."""
        x = x * self.scale
        half_size = self.size // 2
        emb = torch.log(torch.Tensor([10000.0])) / (half_size - 1)
        emb = torch.exp(-emb * torch.arange(half_size))
        emb = x.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat((torch.sin(emb), torch.cos(emb)), dim=-1)
        return emb

    def __len__(self):
        return self.size


class LinearEmbedding(nn.Module):
    """Scalar embedding ``x / size * scale`` (output dimension 1)."""

    def __init__(self, size: int, scale: float = 1.0):
        super().__init__()
        self.size = size
        self.scale = scale

    def forward(self, x: torch.Tensor):
        """Map ``(B,)`` timesteps to ``(B, 1)`` floats."""
        x = x / self.size * self.scale
        return x.unsqueeze(-1)

    def __len__(self):
        return 1


class LearnableEmbedding(nn.Module):
    """Linear projection of the scaled timestep to ``size`` dimensions."""

    def __init__(self, size: int):
        super().__init__()
        self.size = size
        self.linear = nn.Linear(1, size)

    def forward(self, x: torch.Tensor):
        """Map ``(B,)`` timesteps to ``(B, size)`` floats."""
        return self.linear(x.unsqueeze(-1).float() / self.size)

    def __len__(self):
        return self.size


class IdentityEmbedding(nn.Module):
    """Passes the timestep through unchanged as a 1-d feature."""

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        """Map ``(B,)`` timesteps to ``(B, 1)``."""
        return x.unsqueeze(-1)

    def __len__(self):
        return 1


class ZeroEmbedding(nn.Module):
    """Ablation embedding: always zero, so the model sees no time information."""

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        """Return zeros of shape ``(B, 1)``."""
        return x.unsqueeze(-1) * 0

    def __len__(self):
        return 1


class PositionalEmbedding(nn.Module):
    """Factory wrapper around the individual embedding types.

    Parameters
    ----------
    size : int
        Embedding size passed to the chosen embedding.
    type : str
        One of ``sinusoidal``, ``linear``, ``learnable``, ``identity``, ``zero``.
    **kwargs
        Forwarded to the embedding constructor (``scale`` for sinusoidal/linear).
    """

    def __init__(self, size: int, type: str, **kwargs):
        super().__init__()
        self.layer = create_embedding(type, size, **kwargs)

    def forward(self, x: torch.Tensor):
        """Embed timesteps ``x`` and cast the result to float32."""
        return self.layer(x).float()

    def __len__(self):
        # BUG FIX #6: was `return self.size` (undefined) → use layer
        return len(self.layer)


def create_embedding(embedding_type: str, size: int, **kwargs) -> nn.Module:
    """Create an embedding module by type name.

    Raises
    ------
    ValueError
        If ``embedding_type`` is not one of the registered names.
    """
    registry = {
        "sinusoidal": lambda: SinusoidalEmbedding(size, **kwargs),
        "linear": lambda: LinearEmbedding(size, **kwargs),
        "learnable": lambda: LearnableEmbedding(size),
        "identity": lambda: IdentityEmbedding(),
        "zero": lambda: ZeroEmbedding(),
    }
    if embedding_type not in registry:
        raise ValueError(f"Unknown embedding type: {embedding_type}")
    return registry[embedding_type]()
