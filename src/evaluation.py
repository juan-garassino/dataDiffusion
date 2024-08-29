import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import ks_2samp
from sklearn.metrics import mean_squared_error, r2_score

import torch
import torch.nn as nn

from .custom_logger import get_default_logger


logger = get_default_logger()


def evaluate_generative_model(model, X, y, noise_scheduler, verbose=False):
    """
    Evaluates a generative model using a noise scheduler.
    Args: model, X, y, noise_scheduler, verbose (optional)
    Returns: MSE loss and generated samples.
    """
    logger = get_default_logger(verbose)
    logger.info("Evaluating generative model with noise scheduler")
    model.eval()
    with torch.no_grad():
        sample = torch.randn_like(X)
        for t in reversed(range(noise_scheduler.num_timesteps)):
            t_batch = torch.full((X.shape[0],), t, device=X.device, dtype=torch.long)
            predicted_noise = model(sample, t_batch)
            sample = noise_scheduler.step(predicted_noise, t, sample)

        mse = nn.MSELoss()(sample, X)
    logger.info(f"Evaluation MSE: {mse:.4f}")
    logger.info(f"Generated samples shape: {sample.shape}")
    return mse.item(), sample


def evaluate_supervised_model(model, dataloader, verbose=False):
    """
    Evaluates a supervised model using a DataLoader.
    Args: model, dataloader, verbose (optional)
    Returns: MSE and R2 score.
    """
    logger = get_default_logger(verbose)
    logger.info("Evaluating supervised model with dataloader")
    model.eval()
    predictions, targets = [], []
    with torch.no_grad():
        for inputs, batch_targets in dataloader:
            outputs = model(inputs)
            predictions.extend(outputs.numpy().flatten())
            targets.extend(batch_targets.numpy().flatten())

    mse = mean_squared_error(targets, predictions)
    r2 = r2_score(targets, predictions)
    logger.info(f"Evaluation results - MSE: {mse:.4f}, R2: {r2:.4f}")
    return mse, r2


def create_correlation_heatmaps(original_data, generated_data, verbose=False):
    """
    Creates and saves correlation heatmaps for original and generated data.
    Args: original_data, generated_data, verbose (optional)
    """
    logger = get_default_logger(verbose)
    logger.info("Creating correlation heatmaps")
    corr_original = np.corrcoef(original_data.T)
    corr_generated = np.corrcoef(generated_data.T)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    sns.heatmap(corr_original, ax=ax1, cmap="coolwarm", vmin=-1, vmax=1, center=0)
    ax1.set_title("Original Data Correlation")
    sns.heatmap(corr_generated, ax=ax2, cmap="coolwarm", vmin=-1, vmax=1, center=0)
    ax2.set_title("Generated Data Correlation")

    plt.tight_layout()
    plt.savefig("correlation_heatmaps.png")
    plt.close()
    logger.info("Correlation heatmaps saved")


def evaluate_generated_data(original_data, generated_data, verbose=False):
    """
    Evaluates generated data against original data.
    Args: original_data, generated_data, verbose (optional)
    Returns: List of evaluation results for each feature.
    """
    if isinstance(original_data, torch.Tensor):
        original_data = original_data.detach().cpu().numpy()
    if isinstance(generated_data, torch.Tensor):
        generated_data = generated_data.detach().cpu().numpy()

    logger = get_default_logger(verbose)
    logger.info("Evaluating generated data")
    create_correlation_heatmaps(original_data, generated_data, verbose)

    num_features = original_data.shape[1]
    ks_results = []

    for i in range(num_features):
        original_feature = original_data[:, i]
        generated_feature = generated_data[:, i]

        mean_diff = np.abs(np.mean(original_feature) - np.mean(generated_feature))
        std_diff = np.abs(np.std(original_feature) - np.std(generated_feature))
        ks_statistic, p_value = ks_2samp(original_feature, generated_feature)

        ks_results.append(
            {
                "feature": i,
                "mean_diff": mean_diff,
                "std_diff": std_diff,
                "ks_statistic": ks_statistic,
                "p_value": p_value,
            }
        )

        logger.info(f"Feature {i}:")
        logger.info(f"  Mean difference: {mean_diff:.4f}")
        logger.info(f"  Std difference: {std_diff:.4f}")
        logger.info(
            f"  KS test - statistic: {ks_statistic:.4f}, p-value: {p_value:.4f}"
        )

    return ks_results


def print_evaluation_results(ks_results):
    """
    Prints evaluation results for each feature.
    Args: ks_results (list of dictionaries with evaluation metrics)
    """
    print("\nEvaluation Results:")
    print("-------------------")
    for result in ks_results:
        print(f"Feature {result['feature']}:")
        print(f"  KS Statistic: {result['ks_statistic']:.4f}")
        print(f"  p-value: {result['p_value']:.4f}")
        print(f"  Mean Difference: {result['mean_diff']:.4f}")
        print(f"  Std Difference: {result['std_diff']:.4f}")
        print()
