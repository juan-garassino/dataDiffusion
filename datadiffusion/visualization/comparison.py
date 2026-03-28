"""Real vs synthetic distribution comparison plots."""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_marginal_distributions(real, synthetic, feature_names=None, save_path=None):
    """Side-by-side histograms per feature."""
    n_features = real.shape[1]
    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(n_features)]

    cols = 4
    rows = (n_features + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    axes = axes.flatten()

    for i in range(n_features):
        ax = axes[i]
        ax.hist(real[:, i], bins=50, alpha=0.6, label="Real", color="steelblue", density=True)
        ax.hist(synthetic[:, i], bins=50, alpha=0.6, label="Synthetic", color="darkorange", density=True)
        ax.set_title(feature_names[i])
        ax.legend(fontsize=7)

    for j in range(n_features, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_correlation_comparison(real, synthetic, save_path=None):
    """Real vs synthetic correlation heatmaps side by side.

    BUG FIX #10: accepts save_path parameter instead of hardcoding to project root.
    """
    corr_real = np.corrcoef(real.T)
    corr_synth = np.corrcoef(synthetic.T)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    sns.heatmap(corr_real, ax=ax1, cmap="coolwarm", vmin=-1, vmax=1, center=0, square=True)
    ax1.set_title("Real Data Correlation")
    sns.heatmap(corr_synth, ax=ax2, cmap="coolwarm", vmin=-1, vmax=1, center=0, square=True)
    ax2.set_title("Synthetic Data Correlation")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.close(fig)
