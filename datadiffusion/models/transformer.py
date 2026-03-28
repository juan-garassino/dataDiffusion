"""Tabular Transformer denoising model.

BUG FIX #5: forward(x, t) — same signature as MLP. No src_mask argument.
"""

import torch
import torch.nn as nn

from ..embeddings import create_embedding


class TabularTransformer(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
        embedding_type: str = "sinusoidal",
        **embedding_kwargs,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.embedding = nn.Linear(input_size, hidden_size)
        self.time_embedding = create_embedding(embedding_type, hidden_size, **embedding_kwargs)

        encoder_layer = nn.TransformerEncoderLayer(
            hidden_size, num_heads, hidden_size * 4, dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.output_layer = nn.Linear(hidden_size, input_size)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # x: (B, input_size), t: (B,)
        x = self.embedding(x).unsqueeze(1)           # (B, 1, H)
        t_emb = self.time_embedding(t)                 # may be (B, H) or (B, 1, H)
        if t_emb.dim() == 3:
            t_emb = t_emb.squeeze(1)
        t_emb = t_emb.unsqueeze(1)                     # (B, 1, H)
        x = torch.cat([t_emb, x], dim=1)              # (B, 2, H)
        x = self.transformer_encoder(x)               # (B, 2, H)
        x = x[:, 0, :]                                # (B, H)
        return self.output_layer(x)                    # (B, input_size)
