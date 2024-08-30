import torch
import torch.nn as nn
import logging
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
from src.custom_logger import get_default_logger

logger = get_default_logger(verbose=False)

def train_model(model, dataloader, scheduler, num_epochs, learning_rate, verbose=False, model_type="standard", accumulation_steps=1):
    logger.info(f"Starting {model_type} model training")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-6)
    lr_scheduler = OneCycleLR(optimizer, max_lr=learning_rate, epochs=num_epochs, steps_per_epoch=len(dataloader),
                              pct_start=0.3, anneal_strategy="cos", div_factor=10.0, final_div_factor=500.0)

    losses = []

    for epoch in range(num_epochs):
        model.train()
        epoch_losses = []
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for batch_idx, batch in enumerate(progress_bar):
            x, _ = batch
            x = x.to(device)

            if batch_idx % accumulation_steps == 0:
                optimizer.zero_grad()

            timesteps = torch.randint(0, len(scheduler), (x.shape[0],), device=device).long()
            noise = torch.randn_like(x)
            noisy_x = scheduler.add_noise(x, noise, timesteps)

            logger.debug(f"Batch {batch_idx} - noisy_x stats: min={noisy_x.min().item():.4f}, max={noisy_x.max().item():.4f}, mean={noisy_x.mean().item():.4f}, std={noisy_x.std().item():.4f}")

            score = model(noisy_x, timesteps)

            if torch.isnan(score).any() or torch.isinf(score).any():
                logger.warning(f"NaN or Inf detected in score at batch {batch_idx}, epoch {epoch+1}")
                continue

            logger.debug(f"Batch {batch_idx} - score stats: min={score.min().item():.4f}, max={score.max().item():.4f}, mean={score.mean().item():.4f}, std={score.std().item():.4f}")

            target_score = -noise / (scheduler.sqrt_1m_alphas_cumprod[timesteps].view(-1, 1) + 1e-8)

            logger.debug(f"Batch {batch_idx} - target_score stats: min={target_score.min().item():.4f}, max={target_score.max().item():.4f}, mean={target_score.mean().item():.4f}, std={target_score.std().item():.4f}")

            loss = torch.nn.functional.mse_loss(score, target_score)

            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"NaN or Inf loss at batch {batch_idx}, epoch {epoch+1}")
                continue

            loss = loss / accumulation_steps
            loss.backward()

            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                lr_scheduler.step()

            epoch_losses.append(loss.item() * accumulation_steps)
            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})

        avg_loss = sum(epoch_losses) / len(epoch_losses)
        losses.append(avg_loss)
        logger.info(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}, LR: {lr_scheduler.get_last_lr()[0]:.2e}")

    logger.info("Model training completed.")
    return losses
