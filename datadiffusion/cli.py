"""CLI argument parsing → ExperimentConfig."""

import argparse
import os

from .config import (
    ExperimentConfig,
    DataConfig,
    ModelConfig,
    SchedulerConfig,
    TrainingConfig,
    GenerationConfig,
    SelfImprovementConfig,
)


def parse_args(argv=None) -> ExperimentConfig:
    p = argparse.ArgumentParser(description="datadiffusion — tabular diffusion models")

    # Top-level
    p.add_argument("--experiment_name", default="california_housing_diffusion")
    p.add_argument("--dataset", default="california", choices=["california", "moons", "line", "circle"])
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--config", type=str, default=None, help="Path to YAML config file")

    # Data
    p.add_argument("--scaler_type", default="quantile", choices=["quantile", "standard"])

    # Model
    p.add_argument("--model_type", default="mlp", choices=["mlp", "transformer", "score"])
    p.add_argument("--hidden_size", type=int, default=256)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--embedding_type", default="sinusoidal",
                    choices=["sinusoidal", "linear", "learnable", "identity", "zero"])

    # Scheduler
    p.add_argument("--scheduler_type", default="standard", choices=["standard", "score"])
    p.add_argument("--num_timesteps", type=int, default=1000)
    p.add_argument("--beta_start", type=float, default=0.0001)
    p.add_argument("--beta_end", type=float, default=0.02)
    p.add_argument("--beta_schedule", default="cosine", choices=["linear", "quadratic", "cosine"])

    # Training
    p.add_argument("--num_epochs", type=int, default=200)
    p.add_argument("--learning_rate", type=float, default=3e-4)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--accumulation_steps", type=int, default=4)
    p.add_argument("--early_stopping_patience", type=int, default=25)
    p.add_argument("--validation_split", type=float, default=0.1)
    p.add_argument("--importance_sampling", action="store_true", default=True)
    p.add_argument("--no_importance_sampling", action="store_true")

    # Generation
    p.add_argument("--num_samples", type=int, default=1000)
    p.add_argument("--sampler_type", default="ddpm", choices=["ddpm", "ddim"])
    p.add_argument("--ddim_steps", type=int, default=50)

    # Self-improvement
    p.add_argument("--self_improve", action="store_true")
    p.add_argument("--max_iterations", type=int, default=10)
    p.add_argument("--quality_threshold", type=float, default=0.80)

    args = p.parse_args(argv)

    if args.config and os.path.exists(args.config):
        return ExperimentConfig.from_yaml(args.config)

    return ExperimentConfig(
        experiment_name=args.experiment_name,
        dataset=args.dataset,
        verbose=args.verbose,
        data=DataConfig(
            scaler_type=args.scaler_type,
        ),
        model=ModelConfig(
            model_type=args.model_type,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            dropout=args.dropout,
            embedding_type=args.embedding_type,
        ),
        scheduler=SchedulerConfig(
            scheduler_type=args.scheduler_type,
            num_timesteps=args.num_timesteps,
            beta_start=args.beta_start,
            beta_end=args.beta_end,
            beta_schedule=args.beta_schedule,
        ),
        training=TrainingConfig(
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            accumulation_steps=args.accumulation_steps,
            early_stopping_patience=args.early_stopping_patience,
            validation_split=args.validation_split,
            importance_sampling=args.importance_sampling and not args.no_importance_sampling,
        ),
        generation=GenerationConfig(
            num_samples=args.num_samples,
            sampler_type=args.sampler_type,
            ddim_steps=args.ddim_steps,
        ),
        self_improvement=SelfImprovementConfig(
            enabled=args.self_improve,
            max_iterations=args.max_iterations,
            quality_threshold=args.quality_threshold,
        ),
    )
