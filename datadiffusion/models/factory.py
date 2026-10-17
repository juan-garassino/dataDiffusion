"""Model + scheduler factory — breaks circular imports."""

from ..config import ExperimentConfig
from ..schedulers import NoiseScheduler, ScoreBasedNoiseScheduler
from .mlp import TabularMLP
from .transformer import TabularTransformer
from .score import EnhancedScoreNetwork
from ..tracking.logger import get_logger

logger = get_logger("datadiffusion.models")


def create_model_and_scheduler(config: ExperimentConfig, input_size: int):
    mc = config.model
    sc = config.scheduler

    if mc.model_type == "mlp":
        model = TabularMLP(
            input_size=input_size,
            hidden_size=mc.hidden_size,
            hidden_layers=mc.num_layers,
            dropout=mc.dropout,
            embedding_type=mc.embedding_type,
            **mc.embedding_kwargs,
        )
    elif mc.model_type == "transformer":
        model = TabularTransformer(
            input_size=input_size,
            hidden_size=mc.hidden_size,
            num_layers=mc.num_layers,
            num_heads=mc.num_heads,
            dropout=mc.dropout,
            embedding_type=mc.embedding_type,
            **mc.embedding_kwargs,
        )
    elif mc.model_type == "score":
        model = EnhancedScoreNetwork(
            input_dim=input_size,
            hidden_dim=mc.hidden_size,
            num_layers=mc.num_layers,
            dropout=mc.dropout,
            embedding_type=mc.embedding_type,
            num_heads=mc.num_heads,
            **mc.embedding_kwargs,
        )
    else:
        raise ValueError(f"Unknown model type: {mc.model_type}")

    if sc.scheduler_type == "standard":
        scheduler = NoiseScheduler(
            num_timesteps=sc.num_timesteps,
            beta_start=sc.beta_start,
            beta_end=sc.beta_end,
            beta_schedule=sc.beta_schedule,
        )
    elif sc.scheduler_type == "score":
        scheduler = ScoreBasedNoiseScheduler(
            num_timesteps=sc.num_timesteps,
            beta_start=sc.beta_start,
            beta_end=sc.beta_end,
            beta_schedule=sc.beta_schedule,
        )
    else:
        raise ValueError(f"Unknown scheduler type: {sc.scheduler_type}")

    # Score models predict a score; mlp/transformer predict noise. The scheduler's
    # reverse step interprets the model output, so the two must agree.
    if (mc.model_type == "score") != (sc.scheduler_type == "score"):
        logger.warning(
            "model/scheduler mismatch: model_type=%r with scheduler_type=%r; sampling will "
            "misinterpret the model output (use scheduler_type='score' only with model_type='score')",
            mc.model_type, sc.scheduler_type,
        )

    return model, scheduler
