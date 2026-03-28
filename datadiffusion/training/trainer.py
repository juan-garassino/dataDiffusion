"""Trainer with validation loop, early stopping, EMA, and correct loss selection.

BUG FIX #9: On NaN, zero gradients and skip the full accumulation window.
"""

import copy
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm

from ..config import ExperimentConfig
from ..tracking.logger import get_logger
from .losses import get_loss_fn

logger = get_logger("datadiffusion.training")


class EMA:
    """Exponential Moving Average of model parameters."""

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            self.shadow[k].mul_(self.decay).add_(v, alpha=1 - self.decay)

    def apply(self, model):
        """Copy EMA weights into model (call before generation)."""
        self.backup = {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.shadow)

    def restore(self, model):
        """Restore original weights after generation."""
        model.load_state_dict(self.backup)


class Trainer:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.tc = config.training
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ema = None

    def _sample_timesteps(self, batch_size, num_timesteps, device):
        """Sample timesteps, optionally with importance weighting toward low t."""
        if self.tc.importance_sampling:
            weights = 1.0 / (torch.arange(num_timesteps, dtype=torch.float32) + 1.0)
            indices = torch.multinomial(weights, batch_size, replacement=True)
            return indices.to(device)
        return torch.randint(0, num_timesteps, (batch_size,), device=device).long()

    def train(self, model, scheduler, dataset: TensorDataset):
        model.to(self.device)
        loss_fn = get_loss_fn(self.config.model.model_type)

        # Split into train / val
        val_size = max(1, int(len(dataset) * self.tc.validation_split))
        train_size = len(dataset) - val_size
        train_ds, val_ds = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_ds, batch_size=self.tc.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=self.tc.batch_size, shuffle=False)

        optimizer = torch.optim.AdamW(model.parameters(), lr=self.tc.learning_rate, weight_decay=1e-6)
        lr_scheduler = OneCycleLR(
            optimizer,
            max_lr=self.tc.learning_rate,
            epochs=self.tc.num_epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.3,
            anneal_strategy="cos",
            div_factor=10.0,
            final_div_factor=500.0,
        )

        # EMA for smoother generation weights
        self.ema = EMA(model, decay=0.999)

        train_losses, val_losses = [], []
        best_val_loss = float("inf")
        best_ema_shadow = None
        patience_counter = 0

        for epoch in range(self.tc.num_epochs):
            # --- Training ---
            model.train()
            epoch_losses = []
            progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.tc.num_epochs}", leave=False)

            for batch_idx, batch in enumerate(progress):
                x = batch[0].to(self.device)

                if batch_idx % self.tc.accumulation_steps == 0:
                    optimizer.zero_grad()

                timesteps = self._sample_timesteps(x.shape[0], len(scheduler), self.device)
                noise = torch.randn_like(x)
                noisy_x = scheduler.add_noise(x, noise, timesteps)

                predicted = model(noisy_x, timesteps)

                # BUG FIX #9: NaN check — zero grads and skip
                if torch.isnan(predicted).any() or torch.isinf(predicted).any():
                    logger.warning("NaN/Inf in model output at epoch %d batch %d — skipping", epoch + 1, batch_idx)
                    optimizer.zero_grad()
                    continue

                loss_kwargs = dict(predicted=predicted, noise=noise)
                if self.config.model.model_type == "score":
                    loss_kwargs["sqrt_1m_alpha"] = scheduler.sqrt_1m_alphas_cumprod
                    loss_kwargs["timesteps"] = timesteps

                loss = loss_fn(**loss_kwargs)

                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning("NaN/Inf loss at epoch %d batch %d — skipping", epoch + 1, batch_idx)
                    optimizer.zero_grad()
                    continue

                loss = loss / self.tc.accumulation_steps
                loss.backward()

                if (batch_idx + 1) % self.tc.accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    lr_scheduler.step()
                    self.ema.update(model)

                epoch_losses.append(loss.item() * self.tc.accumulation_steps)
                progress.set_postfix({"loss": f"{loss.item():.4f}"})

            avg_train = sum(epoch_losses) / max(len(epoch_losses), 1)
            train_losses.append(avg_train)

            # --- Validation (using EMA weights) ---
            self.ema.apply(model)
            model.eval()
            val_epoch_losses = []
            with torch.no_grad():
                for batch in val_loader:
                    x = batch[0].to(self.device)
                    timesteps = self._sample_timesteps(x.shape[0], len(scheduler), self.device)
                    noise = torch.randn_like(x)
                    noisy_x = scheduler.add_noise(x, noise, timesteps)
                    predicted = model(noisy_x, timesteps)

                    loss_kwargs = dict(predicted=predicted, noise=noise)
                    if self.config.model.model_type == "score":
                        loss_kwargs["sqrt_1m_alpha"] = scheduler.sqrt_1m_alphas_cumprod
                        loss_kwargs["timesteps"] = timesteps

                    val_loss = loss_fn(**loss_kwargs)
                    val_epoch_losses.append(val_loss.item())

            avg_val = sum(val_epoch_losses) / max(len(val_epoch_losses), 1)
            val_losses.append(avg_val)

            # --- Early stopping (on EMA val loss) ---
            if avg_val < best_val_loss:
                best_val_loss = avg_val
                best_ema_shadow = copy.deepcopy(self.ema.shadow)
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.tc.early_stopping_patience:
                    logger.info("Early stopping at epoch %d (patience=%d)", epoch + 1, self.tc.early_stopping_patience)
                    self.ema.restore(model)
                    break

            self.ema.restore(model)

            logger.info(
                "Epoch %d/%d — train_loss=%.4f  val_loss=%.4f  lr=%.2e",
                epoch + 1, self.tc.num_epochs, avg_train, avg_val,
                lr_scheduler.get_last_lr()[0],
            )

        # Load best EMA weights for generation
        if best_ema_shadow is not None:
            model.load_state_dict(best_ema_shadow)

        return train_losses, val_losses
