"""Tabular Transformer denoising model.

BUG FIX #5: forward(x, t) — same signature as MLP. No src_mask argument.
"""

import torch
import torch.nn as nn

from ..embeddings import create_embedding


class TabularTransformer(nn.Module):
    """Two-token transformer encoder for noise prediction.

    The feature row (projected to ``hidden_size``) and the time embedding form
    a sequence of length 2; the output at the time token is projected back to
    ``input_size``.

    Parameters
    ----------
    input_size : int
        Number of features.
    hidden_size : int
        Model width (must be divisible by ``num_heads``).
    num_layers : int
        Number of encoder layers.
    num_heads : int
        Attention heads per layer.
    dropout : float
        Dropout inside the encoder layers.
    embedding_type : str
        Time embedding name, see ``embeddings.create_embedding``.
    **embedding_kwargs
        Forwarded to the embedding constructor.
    """

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
        """Predict noise for ``x`` of shape ``(B, input_size)`` at timesteps ``t`` of shape ``(B,)``."""
        x = self.embedding(x).unsqueeze(1)           # (B, 1, H)
        t_emb = self.time_embedding(t)                 # may be (B, H) or (B, 1, H)
        if t_emb.dim() == 3:
            t_emb = t_emb.squeeze(1)
        t_emb = t_emb.unsqueeze(1)                     # (B, 1, H)
        x = torch.cat([t_emb, x], dim=1)              # (B, 2, H)
        x = self.transformer_encoder(x)               # (B, 2, H)
        x = x[:, 0, :]                                # (B, H)
        return self.output_layer(x)                    # (B, input_size)
