"""Positional / time embeddings for diffusion models."""

import torch
from torch import nn


class SinusoidalEmbedding(nn.Module):
    def __init__(self, size: int, scale: float = 1.0):
        super().__init__()
        self.size = size
        self.scale = scale

    def forward(self, x: torch.Tensor):
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
    def __init__(self, size: int, scale: float = 1.0):
        super().__init__()
        self.size = size
        self.scale = scale

    def forward(self, x: torch.Tensor):
        x = x / self.size * self.scale
        return x.unsqueeze(-1)

    def __len__(self):
        return 1


class LearnableEmbedding(nn.Module):
    def __init__(self, size: int):
        super().__init__()
        self.size = size
        self.linear = nn.Linear(1, size)

    def forward(self, x: torch.Tensor):
        return self.linear(x.unsqueeze(-1).float() / self.size)

    def __len__(self):
        return self.size


class IdentityEmbedding(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        return x.unsqueeze(-1)

    def __len__(self):
        return 1


class ZeroEmbedding(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        return x.unsqueeze(-1) * 0

    def __len__(self):
        return 1


class PositionalEmbedding(nn.Module):
    """Factory wrapper around the individual embedding types."""

    def __init__(self, size: int, type: str, **kwargs):
        super().__init__()
        self.layer = create_embedding(type, size, **kwargs)

    def forward(self, x: torch.Tensor):
        return self.layer(x).float()

    def __len__(self):
        # BUG FIX #6: was `return self.size` (undefined) → use layer
        return len(self.layer)


def create_embedding(embedding_type: str, size: int, **kwargs) -> nn.Module:
    """Create an embedding module by type name."""
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
