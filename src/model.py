import torch
from torch import nn
import numpy as np
import math

from .embedding import (
    SinusoidalEmbedding,
    LinearEmbedding,
    LearnableEmbedding,
    IdentityEmbedding,
    ZeroEmbedding,
)
from .scheduler import NoiseScheduler, ScoreBasedNoiseScheduler
from .custom_logger import get_default_logger
from .data_utils import prepare_data
from .train import train_model
from .evaluation import evaluate_generative_model

logger = get_default_logger()


class TabularMLP(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        hidden_layers: int = 3,
        dropout: float = 0.1,
        embedding_type="sinusoidal",
        **embedding_kwargs,
    ):
        super().__init__()
        self.input_size = input_size
        self.input_norm = nn.BatchNorm1d(input_size)

        if embedding_type == "sinusoidal":
            self.time_embedding = SinusoidalEmbedding(hidden_size, **embedding_kwargs)
        elif embedding_type == "linear":
            self.time_embedding = LinearEmbedding(hidden_size, **embedding_kwargs)
        elif embedding_type == "learnable":
            self.time_embedding = LearnableEmbedding(hidden_size)
        elif embedding_type == "identity":
            self.time_embedding = IdentityEmbedding()
        elif embedding_type == "zero":
            self.time_embedding = ZeroEmbedding()
        else:
            raise ValueError(f"Unknown embedding type: {embedding_type}")

        layers = [
            nn.Linear(input_size + hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        ]
        for _ in range(hidden_layers):
            layers.append(
                nn.Linear(hidden_size, hidden_size)
            )  # Update to use hidden_size consistently
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
        layers.append(
            nn.Linear(hidden_size, input_size)
        )  # Output size should match input size
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(x)
        t = t.unsqueeze(-1).float()
        t_emb = self.time_embedding(t)
        x = torch.cat([x, t_emb], dim=-1)  # Concatenate along the last dimension
        return self.mlp(x)


class TabularTransformer(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size=128,
        num_layers=3,
        num_heads=4,
        dropout=0.1,
        embedding_type="sinusoidal",
        **embedding_kwargs,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.embedding = nn.Linear(input_size, hidden_size)

        if embedding_type == "sinusoidal":
            self.time_embedding = SinusoidalEmbedding(hidden_size, **embedding_kwargs)
        elif embedding_type == "linear":
            self.time_embedding = LinearEmbedding(hidden_size, **embedding_kwargs)
        elif embedding_type == "learnable":
            self.time_embedding = LearnableEmbedding(hidden_size)
        elif embedding_type == "identity":
            self.time_embedding = IdentityEmbedding()
        elif embedding_type == "zero":
            self.time_embedding = ZeroEmbedding()
        else:
            raise ValueError(f"Unknown embedding type: {embedding_type}")

        encoder_layers = nn.TransformerEncoderLayer(
            hidden_size, num_heads, hidden_size * 4, dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)

        self.output_layer = nn.Linear(hidden_size, input_size)

    def forward(self, x, t):
        x = self.embedding(x)  # (batch_size, input_size) -> (batch_size, hidden_size)
        x = x.unsqueeze(1)  # (batch_size, hidden_size) -> (batch_size, 1, hidden_size)

        # Add time embedding
        time_embed = self.time_embedding(t).unsqueeze(
            1
        )  # (batch_size, hidden_size) -> (batch_size, 1, hidden_size)
        x = torch.cat([time_embed, x], dim=1)  # (batch_size, 2, hidden_size)

        # Transformer layers
        x = self.transformer_encoder(
            x
        )  # (batch_size, 2, hidden_size) -> (batch_size, 2, hidden_size)

        # Take the output corresponding to the time token
        x = x[:, 0, :]  # (batch_size, hidden_size)

        # Final output layer
        x = self.output_layer(
            x
        )  # (batch_size, hidden_size) -> (batch_size, input_size)

        return x


class ScoreNetwork(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_layers,
        embedding_type="sinusoidal",
        **embedding_kwargs,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Time embedding
        if embedding_type == "sinusoidal":
            self.time_embed = SinusoidalEmbedding(hidden_dim, **embedding_kwargs)
        elif embedding_type == "linear":
            self.time_embed = LinearEmbedding(hidden_dim, **embedding_kwargs)
        elif embedding_type == "learnable":
            self.time_embed = LearnableEmbedding(hidden_dim)
        elif embedding_type == "identity":
            self.time_embed = IdentityEmbedding()
        elif embedding_type == "zero":
            self.time_embed = ZeroEmbedding()
        else:
            raise ValueError(f"Unknown embedding type: {embedding_type}")

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Main layers
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.Tanh(),
                )
                for _ in range(num_layers)
            ]
        )

        # Output projection
        self.final = nn.Linear(hidden_dim, input_dim)

        # Initialize weights with higher gain
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=1.0)  # Increased gain
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x, t):
        # Normalize input with a higher epsilon to avoid division by zero
        x = (x - x.mean(dim=1, keepdim=True)) / (x.std(dim=1, keepdim=True) + 1e-4)

        # Time embedding
        t = t.view(-1, 1).float() / 1000.0  # Normalize time to [0, 1]
        t_embed = self.time_embed(t)  # (batch_size, hidden_dim)

        # Input projection
        x = self.input_proj(x)  # (batch_size, input_dim) -> (batch_size, hidden_dim)
        x = x + t_embed.unsqueeze(
            1
        )  # (batch_size, hidden_dim) + (batch_size, 1, hidden_dim) -> (batch_size, input_dim, hidden_dim)

        # Main layers
        for layer in self.layers:
            x = layer(
                x
            )  # (batch_size, input_dim, hidden_dim) -> (batch_size, input_dim, hidden_dim)

        # Output projection
        x = self.final(
            x
        )  # (batch_size, input_dim, hidden_dim) -> (batch_size, input_dim, input_dim)

        return x


def create_model_and_scheduler(config, input_size):
    if config.model_type == "mlp":
        model = TabularMLP(
            input_size=input_size,
            hidden_size=config.hidden_size,
            hidden_layers=config.num_layers,
            dropout=config.dropout,
            embedding_type=config.embedding_type,
            **config.embedding_kwargs,
        )
    elif config.model_type == "transformer":
        model = TabularTransformer(
            input_size=input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            dropout=config.dropout,
            embedding_type=config.embedding_type,
            **config.embedding_kwargs,
        )
    elif config.model_type == "score":
        model = ScoreNetwork(
            input_dim=input_size,
            hidden_dim=config.hidden_size,
            num_layers=config.num_layers,
            embedding_type=config.embedding_type,
            **config.embedding_kwargs,
        )
    else:
        raise ValueError(f"Unknown model type: {config.model_type}")

    if config.scheduler_type == "standard":
        noise_scheduler = NoiseScheduler(
            num_timesteps=config.num_timesteps,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
            beta_schedule=config.beta_schedule,
        )
    elif config.scheduler_type == "score":
        noise_scheduler = ScoreBasedNoiseScheduler(
            num_timesteps=config.num_timesteps,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
            beta_schedule=config.beta_schedule,
        )
    else:
        raise ValueError(f"Unknown scheduler type: {config.scheduler_type}")

    return model, noise_scheduler


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()

    logger = get_default_logger(args.verbose)

    # Test the TabularMLP model
    input_size = 8
    model = TabularMLP(input_size)
    print(f"Created TabularMLP with input size: {input_size}")

    # Generate dummy data
    X = torch.randn(100, input_size)
    t = torch.randint(0, 1000, (100,))
    output = model(X, t)
    print(f"Model output shape: {output.shape}")

    # Test data preparation
    X_np = np.random.randn(100, input_size)
    y_np = np.random.randn(100, 1)
    X_tensor, y_tensor = prepare_data(X_np, y_np, verbose=args.verbose)
    print(f"Prepared data shapes - X: {X_tensor.shape}, y: {y_tensor.shape}")

    # Test model training (using a dummy noise scheduler)
    class DummyNoiseScheduler:
        def __init__(self):
            self.num_timesteps = 1000

        def add_noise(self, x, noise, t):
            return x + noise

    scheduler = DummyNoiseScheduler()
    losses = train_model(
        model, X_tensor, y_tensor, scheduler, epochs=5, verbose=args.verbose
    )
    print(f"Training losses: {losses}")

    # Test model evaluation
    mse, samples = evaluate_generative_model(
        model, X_tensor, y_tensor, scheduler, verbose=args.verbose
    )
    print(f"Evaluation MSE: {mse:.4f}")
    print(f"Generated samples shape: {samples.shape}")
