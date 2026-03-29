"""Tests for training — correct loss selection, early stopping, no crash."""

import pytest
import torch

from datadiffusion.training.trainer import Trainer
from datadiffusion.training.losses import get_loss_fn, noise_prediction_loss, score_matching_loss
from datadiffusion.models.factory import create_model_and_scheduler
from datadiffusion.config import ExperimentConfig, TrainingConfig
from datadiffusion.tracking.logger import setup_logging

setup_logging()


def test_loss_selection():
    assert get_loss_fn("mlp") is noise_prediction_loss
    assert get_loss_fn("transformer") is noise_prediction_loss
    assert get_loss_fn("score") is score_matching_loss


def test_mlp_training_no_crash(dummy_8d_data, small_mlp_config):
    model, scheduler = create_model_and_scheduler(small_mlp_config, input_size=8)
    trainer = Trainer(small_mlp_config)
    train_losses, val_losses = trainer.train(model, scheduler, dummy_8d_data)
    assert len(train_losses) > 0
    assert len(val_losses) > 0


def test_transformer_training_no_crash(dummy_8d_data, small_transformer_config):
    model, scheduler = create_model_and_scheduler(small_transformer_config, input_size=8)
    trainer = Trainer(small_transformer_config)
    train_losses, val_losses = trainer.train(model, scheduler, dummy_8d_data)
    assert len(train_losses) > 0


def test_score_training_no_crash(dummy_8d_data, small_score_config):
    model, scheduler = create_model_and_scheduler(small_score_config, input_size=8)
    trainer = Trainer(small_score_config)
    train_losses, val_losses = trainer.train(model, scheduler, dummy_8d_data)
    assert len(train_losses) > 0


def test_importance_sampling_distribution():
    """Importance sampling should skew timesteps toward low values (small t)."""
    config = ExperimentConfig(
        training=TrainingConfig(importance_sampling=True),
    )
    trainer = Trainer(config)
    num_timesteps = 1000
    samples = trainer._sample_timesteps(10000, num_timesteps, torch.device("cpu"))

    # The median should be well below the midpoint (500)
    median = samples.float().median().item()
    assert median < 250, f"Median timestep {median} should be < 250 with importance sampling"

    # Low timesteps (0-99) should have more samples than high timesteps (900-999)
    low_count = (samples < 100).sum().item()
    high_count = (samples >= 900).sum().item()
    assert low_count > high_count * 3, "Low timesteps should be sampled much more often"


def test_uniform_sampling_without_importance():
    """Without importance sampling, timesteps should be roughly uniform."""
    config = ExperimentConfig(
        training=TrainingConfig(importance_sampling=False),
    )
    trainer = Trainer(config)
    samples = trainer._sample_timesteps(10000, 1000, torch.device("cpu"))
    median = samples.float().median().item()
    assert 400 < median < 600, f"Median {median} should be near 500 for uniform sampling"
