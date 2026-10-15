"""Loss curve and training visualisation — no mlflow mixing."""

import os
from typing import Optional, Sequence

import matplotlib.pyplot as plt


def plot_loss_curve(
    train_losses: Sequence[float],
    val_losses: Optional[Sequence[float]] = None,
    save_path: Optional[str] = None,
) -> None:
    """Plot per-epoch training (and optionally validation) loss.

    Parameters
    ----------
    train_losses : sequence of float
        One value per epoch.
    val_losses : sequence of float, optional
        One value per epoch; omitted from the plot when empty or ``None``.
    save_path : str, optional
        If given, the figure is saved there (parent directories are created).
    """
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train")
    if val_losses:
        plt.plot(val_losses, label="Validation")
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path)
    plt.close()
