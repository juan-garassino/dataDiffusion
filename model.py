import torch
from torch import nn
import numpy as np

class Block(nn.Module):
    def __init__(self, size: int):
        super().__init__()
        self.ff = nn.Linear(size, size)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor):
        return x + self.act(self.ff(x))

class TabularMLP(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 128, hidden_layers: int = 3, output_size: int = 1, dropout: float = 0.1):
        super().__init__()
        self.input_size = input_size
        print(f"Initializing TabularMLP with input_size: {input_size}")
        self.input_norm = nn.BatchNorm1d(input_size)
        self.time_mlp = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size)
        )
        layers = [nn.Linear(input_size + hidden_size, hidden_size), nn.GELU(), nn.Dropout(dropout)]
        for _ in range(hidden_layers):
            layers.append(Block(hidden_size))
        layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_size, output_size))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        print(f"Input x shape: {x.shape}")
        print(f"Input t shape: {t.shape}")
        
        if x.dim() != 2 or x.size(1) != self.input_size:
            raise ValueError(f"Expected input shape (batch_size, {self.input_size}), but got {x.shape}")
        
        x = self.input_norm(x)
        print(f"Shape after normalization: {x.shape}")
        
        t = t.unsqueeze(-1).float()
        t_emb = self.time_mlp(t)
        print(f"Shape of time embedding: {t_emb.shape}")
        
        x = torch.cat([x, t_emb], dim=-1)
        print(f"Shape after concatenation: {x.shape}")
        
        return self.mlp(x)

def prepare_data(X, y):
    X_tensor = torch.tensor(X.values if hasattr(X, 'values') else X, dtype=torch.float32)
    y_tensor = torch.tensor(y.values if hasattr(y, 'values') else y, dtype=torch.float32).reshape(-1, 1)
    return X_tensor, y_tensor

def train_model(model, X, y, noise_scheduler, epochs=100, lr=0.001, batch_size=32):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    dataset = torch.utils.data.TensorDataset(X, y)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    losses = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()

            # Sample noise and timesteps
            noise = torch.randn_like(batch_X)
            timesteps = torch.randint(0, noise_scheduler.num_timesteps, (batch_X.shape[0],)).long()

            # Add noise to inputs
            noisy_X = noise_scheduler.add_noise(batch_X, noise, timesteps)

            # Predict noise
            predicted_noise = model(noisy_X, timesteps)

            # Compute loss
            loss = criterion(predicted_noise, noise)

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')

    return losses

def evaluate_model(model, X, y, noise_scheduler):
    model.eval()
    with torch.no_grad():
        # Start from random noise
        sample = torch.randn_like(X)
        for t in reversed(range(noise_scheduler.num_timesteps)):
            t_batch = torch.full((X.shape[0],), t, device=X.device, dtype=torch.long)
            predicted_noise = model(sample, t_batch)
            sample = noise_scheduler.step(predicted_noise, t, sample)

        mse = nn.MSELoss()(sample, X)
    return mse.item(), sample