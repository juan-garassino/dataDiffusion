import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import os
from datetime import datetime
from .custom_logger import get_default_logger
import torch
import mlflow


def ensure_results_folder():
    results_folder = "results"
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    return results_folder


def get_timestamped_filename(base_name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{timestamp}_{base_name}"


def create_plot(data, title, xlabel, ylabel, save_path=None, verbose=False):
    logger = get_default_logger(verbose)
    logger.info(f"Creating plot: {title}")

    plt.figure(figsize=(10, 6))
    plt.plot(data)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if save_path:
        results_folder = ensure_results_folder()
        full_save_path = os.path.join(
            results_folder, get_timestamped_filename(save_path)
        )
        plt.savefig(full_save_path)
        print(f"Plot saved to: {full_save_path}")
        logger.info(f"Plot saved to: {full_save_path}")
    else:
        plt.show()

    plt.close()
    logger.info("Plot creation completed")


def create_animation(data_frames, title, xlabel, ylabel, save_path=None, verbose=False):
    logger = get_default_logger(verbose)
    logger.info(f"Creating animation: {title}")

    # Convert PyTorch tensors to NumPy arrays if necessary
    if isinstance(data_frames[0], torch.Tensor):
        data_frames = [frame.detach().cpu().numpy() for frame in data_frames]

    fig, ax = plt.subplots(figsize=(10, 6))
    (line,) = ax.plot([], [])
    ax.set_xlim(0, len(data_frames[0]))
    ax.set_ylim(
        np.min([np.min(frame) for frame in data_frames]),
        np.max([np.max(frame) for frame in data_frames]),
    )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    def init():
        line.set_data([], [])
        return (line,)

    def animate(i):
        line.set_data(range(len(data_frames[i])), data_frames[i])
        return (line,)

    anim = FuncAnimation(
        fig, animate, init_func=init, frames=len(data_frames), blit=True
    )

    if save_path:
        results_folder = ensure_results_folder()
        full_save_path = os.path.join(
            results_folder, get_timestamped_filename(save_path)
        )
        anim.save(full_save_path, writer="pillow")
        print(f"Animation saved to: {full_save_path}")
        logger.info(f"Animation saved to: {full_save_path}")
    else:
        plt.show()

    plt.close()
    logger.info("Animation creation completed")


def plot_loss_curve(losses, save_path=None, verbose=False):
    logger = get_default_logger(verbose)
    logger.info("Plotting loss curve")

    if save_path is None:
        save_path = "loss_curve.png"

    create_plot(losses, "Training Loss", "Epoch", "Loss", save_path, verbose)

    logger.info("Loss curve plotting completed")


def animate_diffusion_process(samples, save_path=None, verbose=False):
    logger = get_default_logger(verbose)
    logger.info("Animating diffusion process")

    if save_path is None:
        save_path = "diffusion_process.gif"

    create_animation(
        samples, "Diffusion Process", "Feature", "Value", save_path, verbose
    )

    logger.info("Diffusion process animation completed")


def save_results(losses, generated_samples, outdir):
    """
    Save experiment results including loss curve and generated samples.

    Args:
    losses (list): List of loss values during training
    generated_samples (torch.Tensor or np.ndarray): Generated samples
    outdir (str): Output directory path
    """
    # Save loss curve
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title("Training Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    loss_path = os.path.join(outdir, "loss_curve.png")
    plt.savefig(loss_path)
    plt.close()
    mlflow.log_artifact(loss_path)

    # Save generated samples
    if isinstance(generated_samples, torch.Tensor):
        generated_samples = generated_samples.detach().cpu().numpy()

    np.save(os.path.join(outdir, "generated_samples.npy"), generated_samples)
    mlflow.log_artifact(os.path.join(outdir, "generated_samples.npy"))

    # Create and save a plot of generated samples (assuming 2D data for visualization)
    plt.figure(figsize=(10, 10))
    plt.scatter(generated_samples[:, 0], generated_samples[:, 1], alpha=0.5)
    plt.title("Generated Samples")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    samples_plot_path = os.path.join(outdir, "generated_samples_plot.png")
    plt.savefig(samples_plot_path)
    plt.close()
    mlflow.log_artifact(samples_plot_path)

    # Log final loss
    mlflow.log_metric("final_loss", losses[-1])


# Example usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()

    # Example data
    losses = [0.5, 0.4, 0.3, 0.2, 0.1]
    plot_loss_curve(losses, verbose=args.verbose)

    samples = [np.random.randn(100) for _ in range(50)]
    animate_diffusion_process(samples, verbose=args.verbose)
