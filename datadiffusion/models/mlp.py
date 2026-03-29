"""Tabular MLP denoising model with residual connections."""

import torch
import torch.nn as nn

from ..embeddings import create_embedding


class ResidualBlock(nn.Module):
    def __init__(self, size, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(size),
            nn.Linear(size, size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(size, size),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.block(x)


class TabularMLP(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        hidden_layers: int = 3,
        dropout: float = 0.1,
        embedding_type: str = "sinusoidal",
        **embedding_kwargs,
    ):
        super().__init__()
        self.input_size = input_size
        self.time_embedding = create_embedding(embedding_type, hidden_size, **embedding_kwargs)
        emb_size = len(self.time_embedding)

        # Project input + time embedding to hidden size
        self.input_proj = nn.Sequential(
            nn.Linear(input_size + emb_size, hidden_size),
            nn.GELU(),
        )

        # Residual blocks
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_size, dropout) for _ in range(hidden_layers)
        ])

        # Output projection
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, input_size),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embedding(t.unsqueeze(-1).float())
        if t_emb.dim() == 3:
            t_emb = t_emb.squeeze(1)
        h = torch.cat([x, t_emb], dim=-1)
        h = self.input_proj(h)
        for block in self.blocks:
            h = block(h)
        return self.output_proj(h)
