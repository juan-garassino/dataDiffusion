from tqdm import tqdm
import numpy as np
import torch

def generate_diffusion_samples(dataset, model, noise_scheduler, num_samples, num_timesteps=50):
    def get_input_size(dataset):
        return dataset.tensors[0].shape[1] if isinstance(dataset.tensors[0], torch.Tensor) else dataset.tensors[0].size(1)

    input_size = get_input_size(dataset)

    # Forward process
    x0 = dataset.tensors[0][:num_samples]  # Take only the specified number of samples
    forward_samples = [x0]
    for t in range(len(noise_scheduler)):
        timesteps = torch.full((num_samples,), t, dtype=torch.long)
        noise = torch.randn_like(x0)
        sample = noise_scheduler.add_noise(x0, noise, timesteps)
        forward_samples.append(sample)

    # Reverse process
    sample = torch.randn(num_samples, input_size)
    timesteps = list(range(num_timesteps))[::-1]
    reverse_samples = [sample.numpy()]

    for i, t in enumerate(tqdm(timesteps, desc="Generating reverse samples")):
        t = torch.full((num_samples,), t, dtype=torch.long)
        with torch.no_grad():
            residual = model(sample, t)
        sample = noise_scheduler.step(residual, t[0], sample)
        reverse_samples.append(sample.numpy())

    return forward_samples, reverse_samples

# Usage example:
# forward_samples, reverse_samples = generate_diffusion_samples(dataset, model, noise_scheduler, num_samples=1000)

from tqdm import tqdm
import numpy as np
import torch

def generate_samples(model, noise_scheduler, num_samples, input_size, num_timesteps=50):
    """
    Generate samples using only the model and noise scheduler, without a dataset.
    
    Args:
    model: The trained diffusion model
    noise_scheduler: The noise scheduler
    num_samples: Number of samples to generate
    input_size: Number of features in each sample
    num_timesteps: Number of timesteps in the diffusion process
    
    Returns:
    list of numpy arrays: Generated samples at each timestep
    """
    
    # Start with random noise
    sample = torch.randn(num_samples, input_size)
    timesteps = list(range(num_timesteps))[::-1]
    samples = [sample.numpy()]

    for i, t in enumerate(tqdm(timesteps, desc="Generating samples")):
        t = torch.full((num_samples,), t, dtype=torch.long)
        with torch.no_grad():
            residual = model(sample, t)
        sample = noise_scheduler.step(residual, t[0], sample)
        samples.append(sample.numpy())

    return samples

# Usage example:
# model = ...  # Your trained TabularMLP model
# noise_scheduler = ...  # Your noise scheduler
# num_samples = 1000
# input_size = 8  # Number of features in your data
# 
# generated_samples = generate_samples(model, noise_scheduler, num_samples, input_size)