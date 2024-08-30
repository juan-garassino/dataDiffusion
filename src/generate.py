from tqdm import tqdm
import numpy as np
import torch
from .scheduler import NoiseScheduler
from src.scheduler import ScoreBasedNoiseScheduler
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import logging
from .custom_logger import get_default_logger

logger = get_default_logger(verbose=False)

def generate_diffusion_samples(dataset, model, scheduler, num_samples, num_timesteps):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    input_size = dataset.tensors[0].shape[1]

    # Forward process
    x0 = dataset.tensors[0][:num_samples].to(device)
    forward_samples = [x0.cpu().numpy()]

    for t in range(num_timesteps):
        timesteps = torch.full((num_samples,), t, dtype=torch.long, device=device)
        noise = torch.randn_like(x0)
        sample = scheduler.add_noise(x0, noise, timesteps)
        forward_samples.append(sample.cpu().numpy())

    # Reverse process
    sample = torch.randn(num_samples, input_size, device=device)
    reverse_samples = [sample.cpu().numpy()]

    for i in tqdm(range(num_timesteps)[::-1], desc="Generating samples"):
        timesteps = torch.full((num_samples,), i, dtype=torch.long, device=device)
        with torch.no_grad():
            model_output = model(sample, timesteps.float())

        sample = scheduler.step(model_output, timesteps[0], sample)
        reverse_samples.append(sample.cpu().numpy())

    return forward_samples, reverse_samples, device


def generate_samples(model, noise_scheduler, num_samples, input_size, num_timesteps=50):
    # Start with random noise
    sample = torch.randn(num_samples, input_size)
    timesteps = list(range(num_timesteps))[::-1]
    samples = [sample.numpy()]

    for i, t in enumerate(tqdm(timesteps, desc="Generating samples")):
        t = torch.full((num_samples,), t, dtype=torch.long)
        with torch.no_grad():
            residual = model(sample, t)
        sample = noise_scheduler.step(residual, t[0], sample)
        samples.append(sample.numpy())

    return samples


def create_feature_matrix_plot(
    samples, num_features, step, num_timesteps, process_name, max_samples=100
):
    fig, axes = plt.subplots(num_features, num_features, figsize=(15, 15))
    plt.subplots_adjust(hspace=0.5, wspace=0.5)

    samples_to_plot = samples[:max_samples] if len(samples) > max_samples else samples

    for i in range(num_features):
        for j in range(num_features):
            if i < j:  # Upper triangle
                axes[i, j].axis("off")
            elif i == j:  # Diagonal
                axes[i, j].hist(samples_to_plot[:, i], bins=30, alpha=0.7)
                axes[i, j].set_title(f"Feature {i+1}")
            else:  # Lower triangle
                axes[i, j].scatter(
                    samples_to_plot[:, j], samples_to_plot[:, i], alpha=0.5, s=5
                )
                axes[i, j].set_xlabel(f"Feature {j+1}")
                axes[i, j].set_ylabel(f"Feature {i+1}")

    plt.suptitle(
        f"{process_name} - Step {step}/{num_timesteps}\n(Showing up to {len(samples_to_plot)} samples)",
        fontsize=16,
    )

    return fig, axes


def update_plot(
    frame, samples_list, axes, num_features, num_timesteps, process_name, max_samples
):
    samples = samples_list[frame]
    samples_to_plot = samples[:max_samples] if len(samples) > max_samples else samples

    # Clear all axes
    for ax in axes.flatten():
        ax.clear()

    for i in range(num_features):
        for j in range(num_features):
            if i < j:  # Upper triangle
                axes[i, j].axis("off")
            elif i == j:  # Diagonal
                axes[i, j].hist(samples_to_plot[:, i], bins=30, alpha=0.7)
                axes[i, j].set_title(f"Feature {i+1}")
            else:  # Lower triangle
                axes[i, j].scatter(
                    samples_to_plot[:, j], samples_to_plot[:, i], alpha=0.5, s=5
                )
                axes[i, j].set_xlabel(f"Feature {j+1}")
                axes[i, j].set_ylabel(f"Feature {i+1}")

    plt.suptitle(
        f"{process_name} - Step {frame}/{num_timesteps}\n(Showing up to {len(samples_to_plot)} samples)",
        fontsize=16,
    )


def create_animation(samples_list, process_name, max_samples=100, save_dir=None):
    num_features = samples_list[0].shape[1]
    num_timesteps = len(samples_list) - 1

    fig, axes = plt.subplots(num_features, num_features, figsize=(15, 15))
    plt.subplots_adjust(hspace=0.5, wspace=0.5)

    def init():
        for ax in axes.flatten():
            ax.clear()
        return axes.flatten()

    def animate(step):
        update_plot(
            step,
            samples_list,
            axes,
            num_features,
            num_timesteps,
            process_name,
            max_samples,
        )
        return axes.flatten()

    anim = animation.FuncAnimation(
        fig, animate, frames=len(samples_list), init_func=init, interval=200, blit=False
    )

    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        anim.save(
            os.path.join(save_dir, f"{process_name}_animation.mp4"),
            writer="ffmpeg",
            fps=5,
        )

    # Save individual plots
    if save_dir:
        for step in range(0, len(samples_list), 10):  # Save every 10 steps
            fig, _ = create_feature_matrix_plot(
                samples_list[step],
                num_features,
                step,
                num_timesteps,
                process_name,
                max_samples,
            )
            plt.savefig(os.path.join(save_dir, f"{process_name}_step_{step}.png"))
            plt.close(fig)

    return anim


if __name__ == "__main__":
    # Generate some dummy data for testing
    num_samples = 1000
    num_features = 4
    num_timesteps = 50
    samples_list = [
        np.random.randn(num_samples, num_features) for _ in range(num_timesteps)
    ]

    print(
        f"Generated {len(samples_list)} sample sets with shape {samples_list[0].shape}"
    )

    # Create and save animation
    animation = create_animation(
        samples_list, "Test_process", max_samples=100, save_dir="test_plots"
    )
    print("Animation created and saved in 'test_plots' directory")

    # Create and save individual plots
    for step in range(0, num_timesteps, 10):
        fig, _ = create_feature_matrix_plot(
            samples_list[step],
            num_features,
            step,
            num_timesteps,
            "Test_process",
            max_samples=100,
        )
        plt.savefig(f"test_plots/Test_process_step_{step}.png")
        plt.close(fig)
    print("Individual plots saved in 'test_plots' directory")
