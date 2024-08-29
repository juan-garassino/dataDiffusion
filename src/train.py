import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
from src.custom_logger import get_default_logger


def train_model(
    model,
    dataloader,
    scheduler,
    num_epochs,
    learning_rate,
    verbose=False,
    model_type="standard",
    accumulation_steps=1,
):
    logger = get_default_logger(verbose)
    logger.info(f"Starting {model_type} model training")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=1e-6
    )

    # Adjust div_factor and final_div_factor to avoid extreme LR changes
    lr_scheduler = OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        epochs=num_epochs,
        steps_per_epoch=len(dataloader),  # Remove the division by accumulation_steps
        pct_start=0.3,
        anneal_strategy="cos",
        div_factor=10.0,
        final_div_factor=500.0,
    )

    losses = []
    nan_count = 0

    for epoch in range(num_epochs):
        model.train()
        epoch_losses = []
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for batch_idx, batch in enumerate(progress_bar):
            x, _ = batch
            x = x.to(device)

            if batch_idx % accumulation_steps == 0:
                optimizer.zero_grad()

            try:
                timesteps = torch.randint(
                    0, len(scheduler), (x.shape[0],), device=device
                ).long()
                noise = torch.randn_like(x)
                noisy_x = scheduler.add_noise(x, noise, timesteps)

                score = model(noisy_x, timesteps)

                target_score = -noise / (
                    scheduler.sqrt_1m_alphas_cumprod[timesteps].view(-1, 1) + 1e-8
                )

                # Check for NaNs or Infs before loss calculation
                assert torch.isfinite(score).all(), "Score contains NaNs or Infs"
                assert torch.isfinite(
                    target_score
                ).all(), "Target score contains NaNs or Infs"
                assert torch.isfinite(
                    noisy_x
                ).all(), "Noisy input contains NaNs or Infs"

                # Loss calculation
                loss = torch.nn.functional.smooth_l1_loss(score, target_score)

                if not torch.isfinite(loss):
                    nan_count += 1
                    logger.warning(
                        f"Loss is not finite: {loss.item()} (occurred {nan_count} times)"
                    )
                    logger.warning(f"Batch index: {batch_idx}, Epoch: {epoch+1}")
                    continue

                # Normalize loss by accumulation steps
                loss = loss / accumulation_steps
                loss.backward()

                # Clip gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)

                if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(
                    dataloader
                ):
                    optimizer.step()
                    if (batch_idx + 1) % accumulation_steps == 0:
                        lr_scheduler.step()
                    optimizer.zero_grad()  # Clear gradients after accumulation

                epoch_losses.append(
                    loss.item() * accumulation_steps
                )  # Multiply back to log the original scale
                progress_bar.set_postfix(
                    {
                        "loss": f"{loss.item() * accumulation_steps:.4f}",
                        "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                    }
                )
            except RuntimeError as e:
                logger.error(f"Error during training: {str(e)}")
                logger.error(
                    f"x shape: {x.shape}, noisy_x shape: {noisy_x.shape}, timesteps shape: {timesteps.shape}"
                )
                for name, param in model.named_parameters():
                    logger.error(f"{name} shape: {param.shape}")
                continue

        if epoch_losses:
            avg_loss = sum(epoch_losses) / len(epoch_losses)
            losses.append(avg_loss)
            logger.info(
                f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.2e}"
            )
        else:
            logger.warning(f"Epoch {epoch+1}/{num_epochs} had no valid losses")

    logger.info(f"Model training completed. NaN losses occurred {nan_count} times.")
    return losses


# Example usage in main.py:
# losses = train_model(model, dataloader, noise_scheduler, args.num_epochs, args.learning_rate, args.verbose)
