"""Shared fixtures for datadiffusion tests."""

import pytest
import torch
import numpy as np
from torch.utils.data import TensorDataset

from datadiffusion.config import ExperimentConfig, ModelConfig, SchedulerConfig, TrainingConfig, GenerationConfig, SelfImprovementConfig


@pytest.fixture
def dummy_2d_data():
    X = torch.randn(200, 2)
    y = torch.randn(200, 1)
    return TensorDataset(X, y)


@pytest.fixture
def dummy_8d_data():
    X = torch.randn(500, 8)
    y = torch.randn(500, 1)
    return TensorDataset(X, y)


@pytest.fixture
def small_mlp_config():
    return ExperimentConfig(
        model=ModelConfig(model_type="mlp", hidden_size=32, num_layers=1, embedding_type="sinusoidal"),
        scheduler=SchedulerConfig(num_timesteps=50),
        training=TrainingConfig(num_epochs=2, batch_size=64, learning_rate=1e-3, early_stopping_patience=5, validation_split=0.1),
        generation=GenerationConfig(num_samples=50),
        self_improvement=SelfImprovementConfig(enabled=False),
    )


@pytest.fixture
def small_transformer_config():
    return ExperimentConfig(
        model=ModelConfig(model_type="transformer", hidden_size=32, num_layers=1, num_heads=2, embedding_type="sinusoidal"),
        scheduler=SchedulerConfig(num_timesteps=50),
        training=TrainingConfig(num_epochs=2, batch_size=64, learning_rate=1e-3, early_stopping_patience=5, validation_split=0.1),
        generation=GenerationConfig(num_samples=50),
        self_improvement=SelfImprovementConfig(enabled=False),
    )


@pytest.fixture
def small_score_config():
    return ExperimentConfig(
        model=ModelConfig(model_type="score", hidden_size=32, num_layers=1, embedding_type="sinusoidal"),
        scheduler=SchedulerConfig(scheduler_type="score", num_timesteps=50),
        training=TrainingConfig(num_epochs=2, batch_size=64, learning_rate=1e-3, early_stopping_patience=5, validation_split=0.1),
        generation=GenerationConfig(num_samples=50),
        self_improvement=SelfImprovementConfig(enabled=False),
    )
