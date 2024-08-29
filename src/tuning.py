import optuna
import torch
from torch.utils.data import DataLoader
from src.model import TabularTransformer
from src.scheduler import NoiseScheduler
from src.data_loading import get_dataset
from src.evaluation import evaluate_generative_model, evaluate_supervised_model


def objective(trial):
    # Define hyperparameters to optimize
    hidden_size = trial.suggest_int("hidden_size", 64, 256)
    num_layers = trial.suggest_int("num_layers", 2, 6)
    num_heads = trial.suggest_int("num_heads", 2, 8)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-2)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    num_epochs = trial.suggest_int("num_epochs", 50, 200)

    # Load dataset
    dataset = get_dataset("california")
    input_size = dataset.tensors[0].shape[1]

    # Create model and scheduler
    model = TabularTransformer(input_size, hidden_size, num_layers, num_heads, dropout)
    scheduler = NoiseScheduler()

    # Create data loader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Training loop
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        for batch in dataloader:
            x, _ = batch
            optimizer.zero_grad()
            noise = torch.randn_like(x)
            timesteps = torch.randint(0, len(scheduler), (x.shape[0],))
            noisy_x = scheduler.add_noise(x, noise, timesteps)
            predicted_noise = model(noisy_x, timesteps)
            loss = torch.nn.functional.mse_loss(predicted_noise, noise)
            loss.backward()
            optimizer.step()

        # Evaluate model
        mse, _ = evaluate_supervised_model(model, dataset)

        # Report intermediate metric
        trial.report(mse, epoch)

        # Handle pruning based on the intermediate value
        if trial.should_prune():
            raise optuna.TrialPruned()

    return mse


def run_hyperparameter_tuning(n_trials=100):
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    return study.best_params


if __name__ == "__main__":
    best_params = run_hyperparameter_tuning()
    print("Best hyperparameters:", best_params)
