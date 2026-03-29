"""Feature matrix animation for diffusion process visualisation."""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def create_animation(samples_list, process_name, max_samples=100, save_dir=None):
    num_features = samples_list[0].shape[1]
    num_timesteps = len(samples_list) - 1

    fig, axes = plt.subplots(num_features, num_features, figsize=(15, 15))
    plt.subplots_adjust(hspace=0.5, wspace=0.5)

    def animate(step):
        samples = samples_list[step]
        s = samples[:max_samples] if len(samples) > max_samples else samples
        for ax in axes.flatten():
            ax.clear()
        for i in range(num_features):
            for j in range(num_features):
                if i < j:
                    axes[i, j].axis("off")
                elif i == j:
                    axes[i, j].hist(s[:, i], bins=30, alpha=0.7)
                    axes[i, j].set_title(f"F{i+1}")
                else:
                    axes[i, j].scatter(s[:, j], s[:, i], alpha=0.5, s=5)
        plt.suptitle(f"{process_name} — step {step}/{num_timesteps}")

    anim = animation.FuncAnimation(fig, animate, frames=len(samples_list), interval=200, blit=False)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        anim.save(os.path.join(save_dir, f"{process_name}_animation.mp4"), writer="ffmpeg", fps=5)

    plt.close(fig)
    return anim
