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


def test_adjust_config_keeps_hidden_size_divisible_by_num_heads():
    """Repeated capacity bumps must keep hidden_size valid for multi-head attention (G-008)."""
    from datadiffusion.evaluation.composite import QualityMetrics

    config = ExperimentConfig(
        model=ModelConfig(model_type="transformer", hidden_size=256, num_heads=4),
        self_improvement=SelfImprovementConfig(enabled=True, capacity_increase_factor=1.5),
    )
    loop = SelfImprovementLoop(config)
    bad = QualityMetrics(
        feature_reports=[], avg_ks_statistic=0.5, corr_diff_norm=1.0,
        ml_utility_ratio=0.1, r2_real=0.5, r2_synthetic=0.0, composite_score=0.1,
    )
    for _ in range(4):
        loop._adjust_config(bad)
        assert config.model.hidden_size % config.model.num_heads == 0
        assert config.model.hidden_size % 2 == 0
