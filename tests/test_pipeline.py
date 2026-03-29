"""Tests for the self-improvement pipeline."""

import pytest

from datadiffusion.config import ExperimentConfig, ModelConfig, SchedulerConfig, TrainingConfig, GenerationConfig, SelfImprovementConfig
from datadiffusion.pipeline.self_improvement import SelfImprovementLoop
from datadiffusion.tracking.logger import setup_logging

setup_logging()


@pytest.mark.slow
def test_single_iteration_pipeline():
    """Pipeline runs a single iteration end-to-end on California Housing."""
    config = ExperimentConfig(
        dataset="california",
        model=ModelConfig(model_type="mlp", hidden_size=32, num_layers=1),
        scheduler=SchedulerConfig(num_timesteps=20),
        training=TrainingConfig(num_epochs=2, batch_size=256, learning_rate=1e-3, early_stopping_patience=5, validation_split=0.1),
        generation=GenerationConfig(num_samples=100),
        self_improvement=SelfImprovementConfig(enabled=False),
    )
    loop = SelfImprovementLoop(config)
    result = loop.run()
    assert len(result.iterations) == 1
    assert 0.0 <= result.quality.composite_score <= 1.0


@pytest.mark.slow
def test_self_improvement_two_iterations():
    """Self-improvement loop runs 2 iterations, adjusts HPs."""
    config = ExperimentConfig(
        dataset="california",
        model=ModelConfig(model_type="mlp", hidden_size=16, num_layers=1),
        scheduler=SchedulerConfig(num_timesteps=10),
        training=TrainingConfig(num_epochs=2, batch_size=512, learning_rate=1e-3, early_stopping_patience=5, validation_split=0.1),
        generation=GenerationConfig(num_samples=50),
        self_improvement=SelfImprovementConfig(
            enabled=True,
            max_iterations=2,
            quality_threshold=0.99,  # unreachable → forces 2 iterations
        ),
    )
    loop = SelfImprovementLoop(config)
    result = loop.run()
    assert len(result.iterations) == 2
