"""Different methods for positional embeddings. These are not essential for understanding DDPMs, but are relevant for the ablation study."""

import torch
from torch import nn
from torch.nn import functional as F
from .custom_logger import get_default_logger

logger = get_default_logger()


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
    def __init__(self, size: int, type: str, **kwargs):
        super().__init__()
        logger.info(
            f"Initializing PositionalEmbedding with size {size} and type {type}"
        )
        if type == "sinusoidal":
            self.layer = SinusoidalEmbedding(size, **kwargs)
        elif type == "linear":
            self.layer = LinearEmbedding(size, **kwargs)
        elif type == "learnable":
            self.layer = LearnableEmbedding(size)
        elif type == "zero":
            self.layer = ZeroEmbedding()
        elif type == "identity":
            self.layer = IdentityEmbedding()
        else:
            logger.error(f"Unknown positional embedding type: {type}")
            raise ValueError(f"Unknown positional embedding type: {type}")

    def forward(self, x: torch.Tensor):
        return self.layer(x).float()  # Ensure output is float

    def __len__(self):
        return self.size


def test_embeddings(embedding_size, batch_size, max_time, verbose=False):
    logger = get_default_logger(verbose)
    logger.info("Starting embedding tests")
    embedding_types = ["sinusoidal", "linear", "learnable", "zero", "identity"]
    for emb_type in embedding_types:
        logger.info(f"Testing {emb_type} embedding:")
        embedding = PositionalEmbedding(embedding_size, emb_type)

        timesteps = torch.randint(0, max_time, (batch_size,))
        emb_output = embedding(timesteps)

        # Convert to float for mean and std calculations
        emb_output_float = emb_output.float()

        logger.info(f"  Input shape: {timesteps.shape}")
        logger.info(f"  Output shape: {emb_output.shape}")
        logger.info(f"  Output mean: {emb_output_float.mean():.4f}")
        logger.info(f"  Output std: {emb_output_float.std():.4f}")
    logger.info("Embedding tests completed")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()

    test_embeddings(
        embedding_size=32, batch_size=10, max_time=1000, verbose=args.verbose
    )
