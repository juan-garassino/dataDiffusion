import mlflow
from datetime import datetime
import os


def start_run(experiment_name):
    mlflow.set_experiment(experiment_name)
    return mlflow.start_run()


def log_params(params):
    for key, value in params.items():
        mlflow.log_param(key, value)


def log_metrics(metrics):
    for key, value in metrics.items():
        mlflow.log_metric(key, value)


def log_artifact(artifact_path):
    mlflow.log_artifact(artifact_path)


def end_run():
    mlflow.end_run()


def setup_experiment(experiment_name):
    """
    Set up the experiment directory and MLflow tracking.

    Args:
    experiment_name (str): Name of the experiment

    Returns:
    str: Path to the output directory for this experiment run
    """
    # Create a timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Set up the output directory
    outdir = os.path.join("experiments", experiment_name, timestamp)
    os.makedirs(outdir, exist_ok=True)

    # Set up MLflow
    mlflow.set_experiment(experiment_name)
    mlflow.start_run(run_name=timestamp)

    # Log the output directory
    mlflow.log_param("outdir", outdir)

    return outdir
