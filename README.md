# dataDiffusion

Tabular data synthesis using denoising diffusion probabilistic models (DDPM/DDIM). Trains a diffusion model on structured datasets and generates synthetic data that preserves marginal distributions, feature correlations, and downstream ML utility.

## Quickstart

```bash
pip install -r requirements.txt

# Train and evaluate (default: California Housing, cosine schedule, DDIM)
python main.py --num_epochs 50 --num_timesteps 200 --sampler_type ddim

# Quick smoke test
make train-quick

# Run tests
make test
```

## How It Works

The pipeline follows the standard DDPM framework adapted for tabular data:

1. **Forward process** — Gradually add Gaussian noise to real feature vectors over T timesteps
2. **Training** — A residual MLP learns to predict the added noise at each timestep
3. **Reverse process** — Start from pure noise and iteratively denoise to generate new samples
4. **Evaluation** — Compare synthetic vs real data using KS statistics, correlation structure, and ML utility

Key design choices:
- **QuantileTransformer** instead of StandardScaler — preserves heavy tails via rank-based normalization
- **Cosine noise schedule** (Nichol & Dhariwal) — better noise budget distribution than linear
- **DDIM sampling** — deterministic, sharper samples in 50 steps instead of 1000
- **Importance-weighted timesteps** — 1/(t+1) weighting gives more training on fine structure
- **EMA weights** — exponential moving average for smoother generation
- **Residual MLP** with LayerNorm — skip connections for better gradient flow

## CLI Reference

```bash
python main.py [OPTIONS]
```

### Top-level

| Argument | Default | Description |
|---|---|---|
| `--experiment_name` | `california_housing_diffusion` | Name for experiment tracking |
| `--dataset` | `california` | Dataset to use (`california`, `moons`, `line`, `circle`) |
| `--verbose` | `False` | Enable verbose logging |
| `--config` | `None` | Path to YAML config file (overrides CLI args) |

### Data

| Argument | Default | Description |
|---|---|---|
| `--scaler_type` | `quantile` | Feature scaler (`quantile`, `standard`) |

### Model

| Argument | Default | Description |
|---|---|---|
| `--model_type` | `mlp` | Architecture (`mlp`, `transformer`, `score`) |
| `--hidden_size` | `256` | Hidden layer width |
| `--num_layers` | `4` | Number of residual blocks |
| `--num_heads` | `4` | Attention heads (transformer only) |
| `--dropout` | `0.1` | Dropout rate |
| `--embedding_type` | `sinusoidal` | Time embedding (`sinusoidal`, `linear`, `learnable`, `identity`, `zero`) |

### Scheduler

| Argument | Default | Description |
|---|---|---|
| `--scheduler_type` | `standard` | Noise scheduler (`standard`, `score`) |
| `--num_timesteps` | `1000` | Total diffusion timesteps |
| `--beta_start` | `0.0001` | Starting noise level |
| `--beta_end` | `0.02` | Ending noise level |
| `--beta_schedule` | `cosine` | Schedule type (`linear`, `quadratic`, `cosine`) |

### Training

| Argument | Default | Description |
|---|---|---|
| `--num_epochs` | `200` | Maximum training epochs |
| `--learning_rate` | `3e-4` | Peak learning rate (OneCycleLR) |
| `--batch_size` | `64` | Training batch size |
| `--accumulation_steps` | `4` | Gradient accumulation steps |
| `--early_stopping_patience` | `25` | Epochs without improvement before stopping |
| `--validation_split` | `0.1` | Fraction held out for validation |
| `--importance_sampling` | `True` | Weight timestep sampling toward low t |
| `--no_importance_sampling` | — | Disable importance sampling |

### Generation

| Argument | Default | Description |
|---|---|---|
| `--num_samples` | `1000` | Number of synthetic samples to generate |
| `--sampler_type` | `ddpm` | Sampling method (`ddpm`, `ddim`) |
| `--ddim_steps` | `50` | DDIM inference steps (when `sampler_type=ddim`) |

### Self-improvement

| Argument | Default | Description |
|---|---|---|
| `--self_improve` | `False` | Enable iterative train-eval-adjust loop |
| `--max_iterations` | `10` | Maximum improvement iterations |
| `--quality_threshold` | `0.80` | Composite score target to stop early |

## Examples

```bash
# Fast evaluation run
python main.py --num_epochs 50 --num_timesteps 200 \
  --beta_schedule cosine --sampler_type ddim --ddim_steps 50

# Self-improvement loop
python main.py --self_improve --max_iterations 5 --quality_threshold 0.90

# Use standard scaler instead of quantile
python main.py --scaler_type standard

# Transformer architecture with score-based scheduler
python main.py --model_type transformer --scheduler_type score

# YAML config
python main.py --config experiments/my_config.yaml
```

## YAML Configuration

All CLI arguments can be specified in a YAML file:

```yaml
experiment_name: my_experiment
dataset: california
verbose: false

data:
  scaler_type: quantile

model:
  model_type: mlp
  hidden_size: 256
  num_layers: 4
  dropout: 0.1
  embedding_type: sinusoidal

scheduler:
  num_timesteps: 200
  beta_schedule: cosine

training:
  num_epochs: 200
  learning_rate: 3e-4
  batch_size: 64
  importance_sampling: true

generation:
  num_samples: 1000
  sampler_type: ddim
  ddim_steps: 50

self_improvement:
  enabled: false
  max_iterations: 10
  quality_threshold: 0.80
```

## Evaluation Metrics

The quality report evaluates synthetic data on three axes:

| Metric | What it measures | Pass threshold |
|---|---|---|
| **KS statistic** (per feature) | Marginal distribution match | < 0.15 |
| **Frobenius norm** | Correlation matrix similarity | < 0.90 |
| **ML utility ratio** | R² of synthetic-trained model / real-trained model | > 0.80 |

These combine into a **composite score** (0–1). On California Housing, the current pipeline achieves **~0.92** composite.

## Output Artifacts

Each run produces artifacts in `experiments/<name>/<timestamp>/iter_<N>/`:

```
iter_0/
├── loss_curve.png              Training and validation loss
├── marginal_histograms.png     Per-feature distribution comparison
├── correlation_comparison.png  Correlation heatmap comparison
├── quality_report.txt          Detailed quality metrics
└── model_checkpoint.pt         Trained model weights
```

Experiments are also tracked in MLflow (run `make mlflow` to view).

## Project Structure

```
datadiffusion/
├── cli.py                  CLI argument parsing
├── config.py               Dataclass configuration
├── data/
│   ├── loading.py          Dataset loading + QuantileTransformer
│   └── utils.py            Normalize/denormalize utilities
├── embeddings/
│   └── base.py             Time embedding implementations
├── evaluation/
│   ├── composite.py        QualityMetrics aggregation
│   ├── metrics.py          KS, Wasserstein, correlation diff
│   ├── ml_utility.py       Train-on-synthetic ML evaluation
│   └── reports.py          Report formatting
├── generation/
│   ├── sampler.py          DDPM and DDIM sampling
│   └── animation.py        Diffusion process animation
├── models/
│   ├── mlp.py              Residual MLP with LayerNorm
│   ├── transformer.py      Tabular Transformer
│   ├── score.py            Score-based networks
│   └── factory.py          Model + scheduler creation
├── pipeline/
│   └── self_improvement.py Train → eval → adjust loop
├── schedulers/
│   ├── noise_scheduler.py  DDPM scheduler (linear/quadratic/cosine)
│   └── score_scheduler.py  Score-based SDE scheduler
├── tracking/
│   ├── logger.py           Logging setup
│   └── mlflow_utils.py     MLflow experiment tracking
├── training/
│   ├── trainer.py          Training loop with EMA + early stopping
│   └── losses.py           Noise prediction + score matching losses
└── visualization/
    ├── plots.py            Loss curves
    └── comparison.py       Marginal + correlation plots
```

## Make Targets

```bash
make install          # Install dependencies
make train            # Train (100 epochs)
make train-quick      # Quick train (10 epochs)
make evaluate         # Train + evaluate (50 epochs)
make improve          # Self-improvement loop
make test             # Run all tests
make test-fast        # Skip slow tests
make mlflow           # Launch MLflow UI
make clean            # Remove caches and logs
make clean-experiments # Remove all experiment data
```

## Requirements

- Python 3.10+
- PyTorch
- scikit-learn
- MLflow
- matplotlib, seaborn
- See `requirements.txt` for full list
