"""Tests for embedding modules — includes BUG #6 regression."""

import pytest
import torch

from datadiffusion.embeddings import (
    SinusoidalEmbedding,
    LinearEmbedding,
    LearnableEmbedding,
    IdentityEmbedding,
    ZeroEmbedding,
    PositionalEmbedding,
)


@pytest.mark.parametrize("emb_type,expected_len", [
    ("sinusoidal", 32),
    ("linear", 1),
    ("learnable", 32),
    ("identity", 1),
    ("zero", 1),
])
def test_embedding_output_shape(emb_type, expected_len):
    emb = PositionalEmbedding(32, emb_type)
    x = torch.randint(0, 100, (16,))
    out = emb(x)
    assert out.shape[0] == 16
    assert out.shape[-1] == expected_len


def test_positional_embedding_len():
    """Regression test for BUG #6: __len__ referenced undefined self.size."""
    emb = PositionalEmbedding(64, "sinusoidal")
    assert len(emb) == 64

    emb2 = PositionalEmbedding(32, "linear")
    assert len(emb2) == 1


def test_sinusoidal_no_nan():
    emb = SinusoidalEmbedding(64)
    out = emb(torch.arange(100).float())
    assert not torch.isnan(out).any()
