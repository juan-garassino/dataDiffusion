import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from tqdm import tqdm
import argparse

from generate import *
from sourcing import *
from model import *
from scheduler import *

# Assuming you have these already defined/imported
# model (your diffusion model)
# noise_scheduler
# real_world_dataset (your original dataset)

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

    # Define the path to your saved model
    experiment_name = "california_housing_diffusion"

    model_path = f"exps/{experiment_name}/model_final.pth"

    # Load the state dictionary
    state_dict = torch.load(model_path)

    # Apply the loaded state dictionary to your model
    model.load_state_dict(state_dict)

    # Set the model to evaluation mode
    model.eval()

    # Now your model is loaded and ready to use
    print("Model loaded successfully")

    noise_scheduler = NoiseScheduler(
        num_timesteps=config.num_timesteps,
        beta_schedule=config.beta_schedule
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
    )

    # Generate samples
    num_samples = 100000
    input_size = dataset.tensors[0].shape[1]  # Assuming real_world_dataset is a TensorDataset
    
    generated_samples = generate_samples(model, noise_scheduler, num_samples, input_size)
    generated_data = generated_samples[-1]  # Use the final step of the diffusion process

    # Define a simple MLP for regression
    class MLP(nn.Module):
        def __init__(self, input_size, hidden_size=64, output_size=1):
            super(MLP, self).__init__()
            self.layers = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, output_size)
            )
        
        def forward(self, x):
            return self.layers(x)

    # Training function
    def train_model(model, train_loader, val_loader, epochs=100, lr=0.001):
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        for epoch in range(epochs):
            model.train()
            for batch in train_loader:
                inputs, targets = batch
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
            
            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    inputs, targets = batch
                    outputs = model(inputs)
                    val_loss += criterion(outputs, targets).item()
            val_loss /= len(val_loader)
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Val Loss: {val_loss:.4f}')

    # Evaluation function
    def evaluate_model(model, test_loader):
        model.eval()
        predictions = []
        targets = []
        with torch.no_grad():
            for batch in test_loader:
                inputs, batch_targets = batch
                outputs = model(inputs)
                predictions.extend(outputs.numpy().flatten())
                targets.extend(batch_targets.numpy().flatten())
        
        mse = mean_squared_error(targets, predictions)
        r2 = r2_score(targets, predictions)
        return mse, r2

    # Prepare real-world data
    X_real = dataset.tensors[0].numpy()
    y_real = dataset.tensors[1].numpy()

    # Split real-world data
    train_size = int(0.8 * len(X_real))
    X_train, X_test = X_real[:train_size], X_real[train_size:]
    y_train, y_test = y_real[:train_size], y_real[train_size:]

    # Standardize data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    generated_data_scaled = scaler.transform(generated_data)

    # Create DataLoaders
    train_loader_real = DataLoader(TensorDataset(torch.FloatTensor(X_train_scaled), torch.FloatTensor(y_train)), batch_size=32, shuffle=True)
    val_loader_real = DataLoader(TensorDataset(torch.FloatTensor(X_test_scaled), torch.FloatTensor(y_test)), batch_size=32)
    train_loader_generated = DataLoader(TensorDataset(torch.FloatTensor(generated_data_scaled), torch.FloatTensor(y_train)), batch_size=32, shuffle=True)

    # Train and evaluate model on generated data
    model_generated = MLP(input_size)
    train_model(model_generated, train_loader_generated, val_loader_real)
    mse_generated, r2_generated = evaluate_model(model_generated, val_loader_real)

    # Train and evaluate model on real data
    model_real = MLP(input_size)
    train_model(model_real, train_loader_real, val_loader_real)
    mse_real, r2_real = evaluate_model(model_real, val_loader_real)

    print("Model trained on generated data:")
    print(f"MSE: {mse_generated:.4f}, R2: {r2_generated:.4f}")

    print("\nModel trained on real data:")
    print(f"MSE: {mse_real:.4f}, R2: {r2_real:.4f}")