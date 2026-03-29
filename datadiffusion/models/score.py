"""Score-based denoising networks."""

import torch
import torch.nn as nn

from ..embeddings import create_embedding


class ScoreNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float = 0.1,
        embedding_type: str = "sinusoidal",
        **embedding_kwargs,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
            )
            for _ in range(num_layers)
        ])
        self.final = nn.Linear(hidden_dim, input_dim)
        self.time_embedding = create_embedding(embedding_type, hidden_dim, **embedding_kwargs)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=1.0)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    @staticmethod
    def safe_normalize(x, dim=1, eps=1e-8):
        mean = x.mean(dim=dim, keepdim=True)
        var = x.var(dim=dim, keepdim=True, unbiased=False)
        std = torch.sqrt(var + eps)
        normalized = (x - mean) / std
        if torch.isnan(normalized).any() or torch.isinf(normalized).any():
            return x - mean
        return normalized

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x_norm = self.safe_normalize(x)
        t_embed = self.time_embedding(t.float())
        if t_embed.dim() == 3:
            t_embed = t_embed.squeeze(1)
        x = self.input_proj(x_norm)
        x = x + t_embed
        for layer in self.layers:
            x = layer(x) + x
        return self.final(x)


class EnhancedScoreNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float = 0.1,
        embedding_type: str = "sinusoidal",
        use_attention: bool = False,
        num_heads: int = 4,
        **embedding_kwargs,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.time_embedding = create_embedding(embedding_type, hidden_dim, **embedding_kwargs)

        self.layers = nn.ModuleList([
            self._EnhancedBlock(hidden_dim, dropout, use_attention, num_heads)
            for _ in range(num_layers)
        ])
        self.final = nn.Linear(hidden_dim, input_dim)
        self.apply(self._init_weights)

    class _EnhancedBlock(nn.Module):
        def __init__(self, hidden_dim, dropout, use_attention, num_heads):
            super().__init__()
            self.use_attention = use_attention
            self.block = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
            )
            if use_attention:
                self.attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
                self.norm = nn.LayerNorm(hidden_dim)

        def forward(self, x):
            residual = x
            x = self.block(x)
            if self.use_attention:
                attn_output, _ = self.attention(x, x, x)
                x = self.norm(x + attn_output)
            return x + residual

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=1.0)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    @staticmethod
    def safe_normalize(x, dim=1, eps=1e-8):
        mean = x.mean(dim=dim, keepdim=True)
        var = x.var(dim=dim, keepdim=True, unbiased=False)
        return (x - mean) / torch.sqrt(var + eps)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x_norm = self.safe_normalize(x)
        x = self.input_proj(x_norm)
        t_embed = self.time_embedding(t.float())
        if t_embed.dim() == 3:
            t_embed = t_embed.squeeze(1)
        x = x + t_embed
        for layer in self.layers:
            x = layer(x)
        return self.final(x)
