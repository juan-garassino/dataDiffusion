import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import os

def create_feature_matrix_plot(samples, num_features, step, num_timesteps, process_name, max_samples=100):
    fig, axes = plt.subplots(num_features, num_features, figsize=(15, 15))
    plt.subplots_adjust(hspace=0.5, wspace=0.5)

    samples_to_plot = samples[:max_samples] if len(samples) > max_samples else samples

    for i in range(num_features):
        for j in range(num_features):
            if i < j:  # Upper triangle
                axes[i, j].axis('off')
            elif i == j:  # Diagonal
                axes[i, j].hist(samples_to_plot[:, i], bins=30, alpha=0.7)
                axes[i, j].set_title(f'Feature {i+1}')
            else:  # Lower triangle
                axes[i, j].scatter(samples_to_plot[:, j], samples_to_plot[:, i], alpha=0.5, s=5)
                axes[i, j].set_xlabel(f'Feature {j+1}')
                axes[i, j].set_ylabel(f'Feature {i+1}')

    plt.suptitle(f"{process_name} - Step {step}/{num_timesteps}\n(Showing up to {len(samples_to_plot)} samples)", fontsize=16)

    return fig, axes

def update_plot(frame, samples_list, axes, num_features, num_timesteps, process_name, max_samples):
    samples = samples_list[frame]
    samples_to_plot = samples[:max_samples] if len(samples) > max_samples else samples

    # Clear all axes
    for ax in axes.flatten():
        ax.clear()

    for i in range(num_features):
        for j in range(num_features):
            if i < j:  # Upper triangle
                axes[i, j].axis('off')
            elif i == j:  # Diagonal
                axes[i, j].hist(samples_to_plot[:, i], bins=30, alpha=0.7)
                axes[i, j].set_title(f'Feature {i+1}')
            else:  # Lower triangle
                axes[i, j].scatter(samples_to_plot[:, j], samples_to_plot[:, i], alpha=0.5, s=5)
                axes[i, j].set_xlabel(f'Feature {j+1}')
                axes[i, j].set_ylabel(f'Feature {i+1}')

    plt.suptitle(f"{process_name} - Step {frame}/{num_timesteps}\n(Showing up to {len(samples_to_plot)} samples)", fontsize=16)

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
        update_plot(step, samples_list, axes, num_features, num_timesteps, process_name, max_samples)
        return axes.flatten()

    anim = animation.FuncAnimation(fig, animate, frames=len(samples_list),
                                   init_func=init, interval=200, blit=False)

    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        anim.save(os.path.join(save_dir, f'{process_name}_animation.mp4'), writer='ffmpeg', fps=5)

    # Save individual plots
    if save_dir:
        for step in range(0, len(samples_list), 10):  # Save every 10 steps
            fig, _ = create_feature_matrix_plot(samples_list[step], num_features, step, num_timesteps, process_name, max_samples)
            plt.savefig(os.path.join(save_dir, f'{process_name}_step_{step}.png'))
            plt.close(fig)

    return anim

# Usage example:
# forward_samples_list, reverse_samples_list = generate_diffusion_samples(...)
# forward_animation = create_animation(forward_samples_list, "Forward_process", save_dir="forward_plots")
# reverse_animation = create_animation(reverse_samples_list, "Reverse_process", save_dir="reverse_plots")
#
# plt.show()  # If you want to display the animations interactively