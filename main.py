import argparse
import os
import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from src.embedding import PositionalEmbedding
from src.scheduler import NoiseScheduler, ScoreBasedNoiseScheduler
from src.model import create_model_and_scheduler
from src.data_loading import get_dataset
from src.generate import generate_diffusion_samples, create_animation
from src.evaluation import (
    evaluate_generative_model,
    evaluate_supervised_model,
    evaluate_generated_data,
    print_evaluation_results,
)
from src.visualization import plot_loss_curve, save_results
from src.custom_logger import get_default_logger
from src.tuning import run_hyperparameter_tuning
from src.train import train_model
from src.mlflow_utils import setup_experiment

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train a diffusion model")
    parser.add_argument(
        "--experiment_name", type=str, default="california_housing_diffusion"
    )
    parser.add_argument(
        "--dataset", type=str, default="california", choices=["california"]
    )
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=1000)
    parser.add_argument(
        "--num_epochs", type=int, default=100, help="Number of epochs to train"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-6, help="Learning rate"
    )
    parser.add_argument(
        "--hidden_size", type=int, default=32, help="Hidden size of the model"
    )
    parser.add_argument(
        "--num_layers", type=int, default=2, help="Number of layers in the model"
    )
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--save_model_step", type=int, default=50)
    parser.add_argument("--num_timesteps", type=int, default=1000)
    parser.add_argument("--beta_start", type=float, default=0.0001)
    parser.add_argument("--beta_end", type=float, default=0.02)
    parser.add_argument(
        "--beta_schedule", type=str, default="linear", choices=["linear", "quadratic"]
    )
    parser.add_argument(
        "--tune_hyperparameters", action="store_true", help="Run hyperparameter tuning"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["transformer", "mlp", "score"],
        default="transformer",
    )
    parser.add_argument(
        "--scheduler_type", type=str, choices=["standard", "score"], default="standard"
    )
    parser.add_argument("--input_dim", type=int, default=8)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--embedding_type", type=str, default="sinusoidal")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--accumulation_steps", type=int, default=4)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    args = parser.parse_args()

    # Setup logging
    logger = get_default_logger(args.verbose)
    logger.info("Starting main process")

    # Set up experiment
    outdir = setup_experiment(args.experiment_name)

    # Load data
    dataset = get_dataset(args.dataset)
    dataloader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True)
    input_size = dataset.tensors[0].shape[1]

    # Hyperparameter tuning
    if args.tune_hyperparameters:
        logger.info("Running hyperparameter tuning")
        best_params = run_hyperparameter_tuning(dataset)
        # Update args with best parameters
        for key, value in best_params.items():
            setattr(args, key, value)
        logger.info(f"Best hyperparameters: {best_params}")

    # Create model and scheduler
    model, scheduler = create_model_and_scheduler(args, input_size)

    logger.info(f"Model architecture:\n{model}")

    # Train model
    losses = train_model(
        model,
        dataloader,
        scheduler,
        args.num_epochs,
        args.learning_rate,
        verbose=True,
        model_type=args.model_type,
        accumulation_steps=args.accumulation_steps,
    )

    # Save the trained model
    torch.save(model.state_dict(), os.path.join(outdir, "trained_model.pth"))

    # Generate samples
    forward_samples, reverse_samples, device = generate_diffusion_samples(
        dataset, model, scheduler, args.num_samples, args.num_timesteps
    )

    # Convert samples to PyTorch tensors and detach them
    forward_samples = [
        torch.tensor(sample, device=device).detach() for sample in forward_samples
    ]
    reverse_samples = [
        torch.tensor(sample, device=device).detach() for sample in reverse_samples
    ]

    # Further processing
    for t in reversed(range(args.num_timesteps)):
        t_batch = torch.full((args.num_samples,), t, device=device, dtype=torch.long)
        if args.model_type == "transformer":
            src_mask = torch.zeros(
                (forward_samples[-1].shape[1], forward_samples[-1].shape[1]),
                device=device,
            ).bool()
            model_output = model(forward_samples[-1].unsqueeze(2), src_mask).squeeze(2)
        else:
            model_output = model(forward_samples[-1], t_batch.float())

        forward_samples[-1] = scheduler.step(model_output, t, forward_samples[-1])
        reverse_samples[-1] = scheduler.step(model_output, t, reverse_samples[-1])

    # Create animations
    forward_animation = create_animation(
        [sample.cpu().detach().numpy() for sample in forward_samples],
        "Forward_process",
        save_dir=outdir,
    )
    reverse_animation = create_animation(
        [sample.cpu().detach().numpy() for sample in reverse_samples],
        "Reverse_process",
        save_dir=outdir,
    )

    # Evaluate generative model
    logger.info("Evaluating generative model")
    mse, generated_samples = evaluate_generative_model(
        model, dataset.tensors[0], dataset.tensors[1], scheduler, verbose=args.verbose
    )
    logger.info(f"Generative Model Evaluation - MSE: {mse:.4f}")

    # Evaluate supervised model
    logger.info("Evaluating supervised model")
    eval_dataloader = DataLoader(
        dataset, batch_size=args.eval_batch_size, shuffle=False
    )
    mse, r2 = evaluate_supervised_model(model, eval_dataloader, verbose=args.verbose)
    logger.info(f"Supervised Model Evaluation - MSE: {mse:.4f}, R2: {r2:.4f}")

    # Evaluate generated data
    logger.info("Evaluating generated data")
    original_data = dataset.tensors[0]
    generated_data = reverse_samples[-1]
    ks_results = evaluate_generated_data(
        original_data, generated_data, verbose=args.verbose
    )
    print_evaluation_results(ks_results)

    # Save results
    logger.info("Saving results")
    save_results(losses, generated_samples, outdir)

    # Plot loss curve
    plot_loss_curve(losses, save_path=f"{outdir}/loss_plot.png", verbose=args.verbose)

    # Save model
    torch.save(model.state_dict(), f"{outdir}/model_final.pth")

    logger.info("Main process completed")
