"""MLflow tracking helpers."""

import os
from datetime import datetime
from contextlib import contextmanager

import mlflow

from .logger import get_logger

logger = get_logger("datadiffusion.tracking")


def setup_experiment(experiment_name: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join("experiments", experiment_name, timestamp)
    os.makedirs(outdir, exist_ok=True)

    mlflow.set_experiment(experiment_name)
    mlflow.start_run(run_name=timestamp)
    mlflow.log_param("outdir", outdir)
    logger.info("MLflow experiment '%s' started → %s", experiment_name, outdir)
    return outdir


@contextmanager
def nested_run(run_name: str):
    """Context manager for an MLflow nested run."""
    with mlflow.start_run(run_name=run_name, nested=True):
        yield


def log_config(config) -> None:
    """Log an ExperimentConfig (or any dataclass) as MLflow params."""
    from dataclasses import asdict

    flat = _flatten(asdict(config))
    mlflow.log_params(flat)


def _flatten(d: dict, parent_key: str = "", sep: str = ".") -> dict:
    items = {}
    for k, v in d.items():
        key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(_flatten(v, key, sep))
        else:
            items[key] = v
    return items
