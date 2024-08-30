import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from .custom_logger import setup_logger
import mlflow
from .data_utils import normalize_data, denormalize_data, prepare_data

logger = setup_logger("scheduler", "logs/scheduler.log")

class NoiseScheduler:
    def __init__(
        self,
        num_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="linear",
    ):
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_schedule = beta_schedule

        # Create the beta schedule
        if beta_schedule == "linear":
            self.betas = torch.linspace(
                beta_start, beta_end, num_timesteps, dtype=torch.float32
            )
        elif beta_schedule == "quadratic":
            self.betas = (
                torch.linspace(
                    beta_start**0.5, beta_end**0.5, num_timesteps, dtype=torch.float32
                )
                ** 2
            )

        # Calculate alphas and related values
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        # Pre-calculate values required for adding noise
        self.sqrt_alphas_cumprod = self.alphas_cumprod**0.5
        self.sqrt_one_minus_alphas_cumprod = (1 - self.alphas_cumprod) ** 0.5

        # Pre-calculate values required for reconstructing x0
        self.sqrt_inv_alphas_cumprod = torch.sqrt(1 / self.alphas_cumprod)
        self.sqrt_inv_alphas_cumprod_minus_one = torch.sqrt(1 / self.alphas_cumprod - 1)

        # Pre-calculate values required for q_posterior
        self.posterior_mean_coef1 = (
            self.betas
            * torch.sqrt(self.alphas_cumprod_prev)
            / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * torch.sqrt(self.alphas)
            / (1.0 - self.alphas_cumprod)
        )

        logger.info(f"Initialized NoiseScheduler with {num_timesteps} timesteps")
        self.log_parameters()

    def log_parameters(self):
        params = {
            "num_timesteps": self.num_timesteps,
            "beta_start": self.beta_start,
            "beta_end": self.beta_end,
            "beta_schedule": self.beta_schedule,
        }
        mlflow.log_params(params)

    def reconstruct_x0(self, x_t, t, noise):
        """
        Reconstruct the original sample from a noisy sample and predicted noise.
        """
        s1 = self.sqrt_inv_alphas_cumprod[t].view(-1, 1)
        s2 = self.sqrt_inv_alphas_cumprod_minus_one[t].view(-1, 1)
        return s1 * x_t - s2 * noise

    def q_posterior(self, x_0, x_t, t):
        """
        Compute the mean of the posterior distribution q(x_{t-1} | x_t, x_0).
        """
        s1 = self.posterior_mean_coef1[t].view(-1, 1)
        s2 = self.posterior_mean_coef2[t].view(-1, 1)
        return s1 * x_0 + s2 * x_t

    def get_variance(self, t):
        """
        Compute the variance for the posterior distribution at timestep t.
        """
        if t == 0:
            return 0
        variance = (
            self.betas[t]
            * (1.0 - self.alphas_cumprod_prev[t])
            / (1.0 - self.alphas_cumprod[t])
        )
        return variance.clamp(min=1e-20)  # Ensure variance is not too small

    def step(self, model_output, timestep, sample):
        """
        Perform one step of the reverse diffusion process.
        """
        t = timestep
        pred_original_sample = self.reconstruct_x0(sample, t, model_output)
        pred_prev_sample = self.q_posterior(pred_original_sample, sample, t)

        variance = 0
        if t > 0:
            noise = torch.randn_like(model_output)
            variance = (self.get_variance(t) ** 0.5) * noise

        pred_prev_sample = pred_prev_sample + variance
        return pred_prev_sample

    def add_noise(self, x_start, x_noise, timesteps):
        """
        Add noise to the original sample to get a noisy sample at a specified timestep.
        """
        s1 = self.sqrt_alphas_cumprod[timesteps].view(-1, 1)
        s2 = self.sqrt_one_minus_alphas_cumprod[timesteps].view(-1, 1)
        return s1 * x_start + s2 * x_noise

    def __len__(self):
        """
        Get the number of timesteps in the diffusion process.
        """
        return self.num_timesteps

    def prepare_data(self, data):
        """
        Prepare the input data for the diffusion process.
        """
        return prepare_data(data)

    def normalize_data(self, data):
        """
        Normalize the input data to have zero mean and unit variance.
        """
        normalized_data, self.data_mean, self.data_std = normalize_data(data)
        return normalized_data

    def denormalize_data(self, normalized_data):
        """
        Denormalize the data using stored mean and standard deviation.
        """
        return denormalize_data(normalized_data, self.data_mean, self.data_std)


class ScoreBasedNoiseScheduler(NoiseScheduler):
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02, beta_schedule="linear"):
        super().__init__(num_timesteps, beta_start, beta_end, beta_schedule)
        self.sqrt_alphas_cumprod = torch.sqrt(torch.cumprod(1 - self.betas, dim=0))
        self.sqrt_1m_alphas_cumprod = torch.sqrt(1 - torch.cumprod(1 - self.betas, dim=0))

    def add_noise(self, x_start, noise, timesteps):
        sqrt_alphas_cumprod = self.sqrt_alphas_cumprod[timesteps].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod = self.sqrt_1m_alphas_cumprod[timesteps].view(-1, 1, 1, 1)
        return sqrt_alphas_cumprod * x_start + sqrt_one_minus_alphas_cumprod * noise

    def step(self, score, t, x):
        # Euler-Maruyama step for reverse-time SDE
        dt = -1 / self.num_timesteps
        z = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
        drift = -0.5 * self.betas[t] * x - self.betas[t] * score
        diffusion = torch.sqrt(self.betas[t].clamp(min=1e-5))
        x_prev = x + drift * dt + diffusion * np.sqrt(abs(dt)) * z
        return x_prev
    
    


if __name__ == "__main__":
    # Start an MLflow run
    with mlflow.start_run():
        scheduler = NoiseScheduler(num_timesteps=1000, beta_start=0.0001, beta_end=0.02)
        logger.info(f"Number of timesteps: {len(scheduler)}")

        x_start = torch.randn(10, 5)
        x_noise = torch.randn_like(x_start)
        timesteps = torch.randint(0, len(scheduler), (10,))
        x_t = scheduler.add_noise(x_start, x_noise, timesteps)
        logger.info(f"Shape of noisy samples: {x_t.shape}")

        model_output = torch.randn_like(x_start)
        x_t_minus_1 = scheduler.step(model_output, 500, x_t)
        logger.info(f"Shape of denoised samples: {x_t_minus_1.shape}")

        data = np.random.randn(100, 5)
        normalized_data, mean, std = scheduler.normalize_data(data)
        logger.info(f"Mean of normalized data: {normalized_data.mean():.4f}")
        logger.info(f"Std of normalized data: {normalized_data.std():.4f}")

        mlflow.log_metrics(
            {
                "normalized_data_mean": normalized_data.mean(),
                "normalized_data_std": normalized_data.std(),
            }
        )
