"""Nested dataclass configuration for datadiffusion experiments."""

from dataclasses import dataclass, field, asdict
from typing import Optional
import yaml


@dataclass
class DataConfig:
    scaler_type: str = "quantile"  # quantile | standard


@dataclass
class ModelConfig:
    model_type: str = "mlp"  # mlp | transformer | score
    hidden_size: int = 256
    num_layers: int = 4
    num_heads: int = 4
    dropout: float = 0.1
    embedding_type: str = "sinusoidal"
    embedding_kwargs: dict = field(default_factory=dict)


@dataclass
class SchedulerConfig:
    scheduler_type: str = "standard"  # standard | score
    num_timesteps: int = 1000
    beta_start: float = 0.0001
    beta_end: float = 0.02
    beta_schedule: str = "cosine"  # linear | quadratic | cosine


@dataclass
class TrainingConfig:
    num_epochs: int = 200
    learning_rate: float = 3e-4
    batch_size: int = 64
    accumulation_steps: int = 4
    early_stopping_patience: int = 25
    validation_split: float = 0.1
    importance_sampling: bool = True


@dataclass
class GenerationConfig:
    num_samples: int = 1000
    sampler_type: str = "ddpm"  # ddpm | ddim
    ddim_steps: int = 50


@dataclass
class SelfImprovementConfig:
    enabled: bool = False
    max_iterations: int = 10
    quality_threshold: float = 0.80
    convergence_delta: float = 0.01
    lr_decay_factor: float = 0.5
    epoch_increase_factor: float = 1.5
    capacity_increase_factor: float = 1.5


@dataclass
class ExperimentConfig:
    experiment_name: str = "california_housing_diffusion"
    dataset: str = "california"
    verbose: bool = False
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    self_improvement: SelfImprovementConfig = field(default_factory=SelfImprovementConfig)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_yaml(cls, path: str) -> "ExperimentConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(
            experiment_name=data.get("experiment_name", "california_housing_diffusion"),
            dataset=data.get("dataset", "california"),
            verbose=data.get("verbose", False),
            data=DataConfig(**data.get("data", {})),
            model=ModelConfig(**data.get("model", {})),
            scheduler=SchedulerConfig(**data.get("scheduler", {})),
            training=TrainingConfig(**data.get("training", {})),
            generation=GenerationConfig(**data.get("generation", {})),
            self_improvement=SelfImprovementConfig(**data.get("self_improvement", {})),
        )
