"""Loss curve and training visualisation — no mlflow mixing."""

import os
import matplotlib.pyplot as plt


def plot_loss_curve(train_losses, val_losses=None, save_path=None):
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
