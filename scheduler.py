import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd

class NoiseScheduler:
    """
    A noise scheduler for diffusion models, adapted for multidimensional data.
    This class manages the noise schedule and provides methods for the forward and reverse processes.
    """

    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02, beta_schedule="linear"):
        """
        Initialize the NoiseScheduler.

        Args:
            num_timesteps (int): The number of timesteps in the diffusion process.
            beta_start (float): The starting value for the noise schedule.
            beta_end (float): The ending value for the noise schedule.
            beta_schedule (str): The type of schedule for beta values ("linear" or "quadratic").
        """
        self.num_timesteps = num_timesteps

        # Create the beta schedule
        if beta_schedule == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float32)
        elif beta_schedule == "quadratic":
            self.betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_timesteps, dtype=torch.float32) ** 2

        # Calculate alphas and related values
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.)

        # Pre-calculate values required for adding noise
        self.sqrt_alphas_cumprod = self.alphas_cumprod ** 0.5
        self.sqrt_one_minus_alphas_cumprod = (1 - self.alphas_cumprod) ** 0.5

        # Pre-calculate values required for reconstructing x0
        self.sqrt_inv_alphas_cumprod = torch.sqrt(1 / self.alphas_cumprod)
        self.sqrt_inv_alphas_cumprod_minus_one = torch.sqrt(1 / self.alphas_cumprod - 1)

        # Pre-calculate values required for q_posterior
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1. - self.alphas_cumprod)

    def reconstruct_x0(self, x_t, t, noise):
        """
        Reconstruct the original sample from a noisy sample and predicted noise.

        Args:
            x_t (torch.Tensor): The noisy sample at timestep t. Shape: (batch_size, num_features)
            t (torch.Tensor): The current timestep. Shape: (batch_size,)
            noise (torch.Tensor): The predicted noise. Shape: (batch_size, num_features)

        Returns:
            torch.Tensor: The reconstructed original sample. Shape: (batch_size, num_features)
        """
        s1 = self.sqrt_inv_alphas_cumprod[t].view(-1, 1)
        s2 = self.sqrt_inv_alphas_cumprod_minus_one[t].view(-1, 1)
        return s1 * x_t - s2 * noise

    def q_posterior(self, x_0, x_t, t):
        """
        Compute the mean of the posterior distribution q(x_{t-1} | x_t, x_0).

        Args:
            x_0 (torch.Tensor): The predicted original sample. Shape: (batch_size, num_features)
            x_t (torch.Tensor): The noisy sample at timestep t. Shape: (batch_size, num_features)
            t (torch.Tensor): The current timestep. Shape: (batch_size,)

        Returns:
            torch.Tensor: The mean of the posterior distribution. Shape: (batch_size, num_features)
        """
        s1 = self.posterior_mean_coef1[t].view(-1, 1)
        s2 = self.posterior_mean_coef2[t].view(-1, 1)
        return s1 * x_0 + s2 * x_t

    def get_variance(self, t):
        """
        Compute the variance for the posterior distribution at timestep t.

        Args:
            t (int): The current timestep.

        Returns:
            float: The variance for the posterior distribution.
        """
        if t == 0:
            return 0

        variance = self.betas[t] * (1. - self.alphas_cumprod_prev[t]) / (1. - self.alphas_cumprod[t])
        return variance.clamp(min=1e-20)  # Ensure variance is not too small

    def step(self, model_output, timestep, sample):
        """
        Perform one step of the reverse diffusion process.

        Args:
            model_output (torch.Tensor): The output from the model (predicted noise). Shape: (batch_size, num_features)
            timestep (int): The current timestep.
            sample (torch.Tensor): The current noisy sample. Shape: (batch_size, num_features)

        Returns:
            torch.Tensor: The sample for the previous timestep. Shape: (batch_size, num_features)
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

        Args:
            x_start (torch.Tensor): The original, noiseless sample. Shape: (batch_size, num_features)
            x_noise (torch.Tensor): The noise to be added. Shape: (batch_size, num_features)
            timesteps (torch.Tensor): The timesteps at which to add noise. Shape: (batch_size,)

        Returns:
            torch.Tensor: The noisy sample. Shape: (batch_size, num_features)
        """
        s1 = self.sqrt_alphas_cumprod[timesteps].view(-1, 1)
        s2 = self.sqrt_one_minus_alphas_cumprod[timesteps].view(-1, 1)

        return s1 * x_start + s2 * x_noise

    def __len__(self):
        """
        Get the number of timesteps in the diffusion process.

        Returns:
            int: The number of timesteps.
        """
        return self.num_timesteps

    @staticmethod
    def normalize_data(data):
        """
        Normalize the input data to have zero mean and unit variance.

        Args:
            data (numpy.ndarray or pandas.DataFrame): Input data.

        Returns:
            tuple: Normalized data, mean, and standard deviation.
        """
        if isinstance(data, pd.DataFrame):
            mean = data.mean()
            std = data.std()
            normalized_data = (data - mean) / std
        else:
            mean = np.mean(data, axis=0)
            std = np.std(data, axis=0)
            normalized_data = (data - mean) / std

        return normalized_data, mean, std

    @staticmethod
    def denormalize_data(normalized_data, mean, std):
        """
        Denormalize the data using stored mean and standard deviation.

        Args:
            normalized_data (numpy.ndarray or pandas.DataFrame): Normalized data.
            mean (numpy.ndarray or pandas.Series): Mean used for normalization.
            std (numpy.ndarray or pandas.Series): Standard deviation used for normalization.

        Returns:
            numpy.ndarray or pandas.DataFrame: Denormalized data.
        """
        return normalized_data * std + mean

    def prepare_data(self, data):
        """
        Prepare the input data for the diffusion process.

        Args:
            data (numpy.ndarray or pandas.DataFrame): Input data.

        Returns:
            torch.Tensor: Prepared data as a PyTorch tensor.
        """
        normalized_data, self.data_mean, self.data_std = self.normalize_data(data)
        return torch.FloatTensor(normalized_data.values if isinstance(normalized_data, pd.DataFrame) else normalized_data)