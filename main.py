import argparse
import os
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import numpy as np

from embedding import *
from scheduler import *
from train import *
from sourcing import *
from animate import *
from generate import *

import matplotlib.pyplot as plt
import numpy as np

#from tabular_mlp import TabularMLP
#from dataset_loader import get_dataset
#from noise_scheduler import NoiseScheduler  # Import the NoiseScheduler

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, default="california_housing_diffusion")
    parser.add_argument("--dataset", type=str, default="california", choices=["california"])
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=1000)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--hidden_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--save_model_step", type=int, default=50)
    parser.add_argument("--num_timesteps", type=int, default=30)
    parser.add_argument("--beta_schedule", type=str, default="linear", choices=["linear", "quadratic"])

    config, _ = parser.parse_known_args()

    dataset = get_dataset(config.dataset)

    dataloader = DataLoader(
        dataset, batch_size=config.train_batch_size, shuffle=True, drop_last=True)

    model = TabularMLP(
        input_size=dataset.tensors[0].shape[1],
        hidden_size=config.hidden_size,
        hidden_layers=config.hidden_layers,
        dropout=config.dropout
    )

    noise_scheduler = NoiseScheduler(
        num_timesteps=config.num_timesteps,
        beta_schedule=config.beta_schedule
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
    )

    global_step = 0
    losses = []
    print("Training model...")
    for epoch in range(config.num_epochs):
        model.train()
        progress_bar = tqdm(total=len(dataloader))
        progress_bar.set_description(f"Epoch {epoch}")
        for step, (batch_X, batch_y) in enumerate(dataloader):
            batch = batch_X  # We'll use batch_X as our data
            noise = torch.randn(batch.shape)
            timesteps = torch.randint(
                0, noise_scheduler.num_timesteps, (batch.shape[0],)
            ).long()

            noisy = noise_scheduler.add_noise(batch, noise, timesteps)
            noise_pred = model(noisy, timesteps)
            loss = F.mse_loss(noise_pred, noise)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "step": global_step}
            losses.append(loss.detach().item())
            progress_bar.set_postfix(**logs)
            global_step += 1
        progress_bar.close()

        if (epoch + 1) % config.save_model_step == 0 or epoch == config.num_epochs - 1:
            print(f"Saving model at epoch {epoch + 1}...")
            outdir = f"exps/{config.experiment_name}"
            os.makedirs(outdir, exist_ok=True)
            torch.save(model.state_dict(), f"{outdir}/model_epoch_{epoch + 1}.pth")

    print("Saving final model...")
    outdir = f"exps/{config.experiment_name}"
    os.makedirs(outdir, exist_ok=True)
    torch.save(model.state_dict(), f"{outdir}/model_final.pth")

    print("Generating samples...")
    model.eval()
    sample = torch.randn(config.eval_batch_size, dataset.tensors[0].shape[1])
    timesteps = list(range(noise_scheduler.num_timesteps))[::-1]
    for i, t in enumerate(tqdm(timesteps)):
        t = torch.from_numpy(np.repeat(t, config.eval_batch_size)).long()
        with torch.no_grad():
            residual = model(sample, t)
        sample = noise_scheduler.step(residual, t[0], sample)

    print("Saving loss plot...")
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Step')
    plt.ylabel('MSE Loss')
    plt.savefig(f"{outdir}/loss_plot.png")
    plt.close()

    print("Saving loss as numpy array...")
    np.save(f"{outdir}/loss.npy", np.array(losses))

    print("Saving generated samples...")
    np.save(f"{outdir}/generated_samples.npy", sample.numpy())

    print("Training complete!")

    ######################################

    #num_features = 8 #samples[0].shape[1]  # Get the number of features from the first sample

    forward_samples, reverse_samples = generate_diffusion_samples(dataset, model, noise_scheduler, 120, num_timesteps=config.num_timesteps)

    forward_animation = create_animation(forward_samples, "Forward_process", save_dir="forward_plots")

    reverse_animation = create_animation(reverse_samples, "Reverse_process", save_dir="reverse_plots")