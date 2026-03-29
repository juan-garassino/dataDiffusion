# Architecture

## Pipeline Overview

```
┌─────────────┐    ┌──────────────┐    ┌────────────┐    ┌────────────┐
│  Load Data  │───▶│    Train     │───▶│  Generate   │───▶│  Evaluate  │
│  + Scale    │    │  (Denoise)   │    │  (Reverse)  │    │  (Compare) │
└─────────────┘    └──────────────┘    └────────────┘    └────────────┘
       │                  │                   │                  │
  QuantileTransformer  Residual MLP      DDPM / DDIM      KS + Corr +
  preserves tails      predicts noise    reverse process   ML Utility
```

The self-improvement loop wraps this pipeline, adjusting hyperparameters between iterations based on quality diagnostics.

## Diffusion Process

### Forward Process (Adding Noise)

Given clean data x₀, the forward process produces a sequence of increasingly noisy versions:

```
x_t = √(ᾱ_t) · x₀ + √(1 - ᾱ_t) · ε     where ε ~ N(0, I)
```

- `ᾱ_t = ∏ᵢ₌₁ᵗ αᵢ` is the cumulative product of `αᵢ = 1 - βᵢ`
- `βᵢ` follows the chosen schedule (cosine by default)
- At t=0, x_t ≈ x₀ (clean). At t=T, x_t ≈ N(0, I) (pure noise)

### Reverse Process (Denoising)

The model learns to predict the noise ε given (x_t, t):

```
ε_θ(x_t, t) ≈ ε
```

**DDPM sampling** reconstructs x₀ from the prediction, computes the posterior mean, and adds noise:

```
x_{t-1} = posterior_mean(x_t, ε_θ) + σ_t · z    where z ~ N(0, I)
```

**DDIM sampling** uses a deterministic update rule (eta=0) over a subsequence of timesteps:

```
x_{t-1} = √(ᾱ_{t-1}) · x̂₀ + √(1 - ᾱ_{t-1}) · ε_θ
```

DDIM produces sharper samples and runs in 50 steps instead of 1000.

## Noise Schedules

### Linear
```
β_t = β_start + t/(T-1) · (β_end - β_start)
```

### Cosine (Nichol & Dhariwal, 2021)
```
ᾱ_t = cos²((t/T + s) / (1 + s) · π/2)     where s = 0.008
β_t = 1 - ᾱ_t / ᾱ_{t-1}                    clamped to [0, 0.999]
```

The cosine schedule distributes the noise budget more evenly across timesteps, spending more capacity on the crucial middle range where structure is resolved.

## Model Architecture

### Residual MLP (Default)

```
Input (B, 8) ──┐
               ├── concat ── Linear ── GELU ──┐
Time (B,) ─ SinEmb(256) ─┘                    │
                                               ▼
                                    ┌─── ResidualBlock ×4 ───┐
                                    │  LayerNorm              │
                                    │  Linear → GELU → Drop   │
                                    │  Linear → GELU → Drop   │
                                    │  + skip connection       │
                                    └─────────────────────────┘
                                               │
                                     LayerNorm ── Linear ── Output (B, 8)
```

Key design choices:
- **No BatchNorm on input** — BatchNorm re-normalizes the noisy signal, destroying the noise the model needs to predict
- **Residual connections** — better gradient flow and preserves input information
- **LayerNorm** instead of BatchNorm — stable with varying noise levels
- **GELU activation** — smoother gradients than ReLU

### Time Embedding

Sinusoidal positional embedding (default) maps scalar timestep t to a 256-dimensional vector:

```
emb[2i]   = sin(t / 10000^(2i/d))
emb[2i+1] = cos(t / 10000^(2i/d))
```

This gives the model a rich representation of "how noisy is the input" at each timestep.

## Training

### Importance-Weighted Timestep Sampling

Instead of uniform sampling over [0, T), timesteps are sampled with probability proportional to 1/(t+1):

```
P(t) ∝ 1/(t+1)
```

This gives ~7x more training on low-noise timesteps (small t), where fine structure — tails, sharp modes, correlations — is resolved. Uniform sampling wastes half the compute on high-noise timesteps where the model only needs to predict "roughly Gaussian noise."

### EMA (Exponential Moving Average)

The trainer maintains a shadow copy of model weights updated after each optimizer step:

```
θ_ema ← 0.999 · θ_ema + 0.001 · θ
```

The EMA weights are used for:
- Validation loss computation (smoother signal for early stopping)
- Final model used for generation

This produces more stable, higher-quality samples than using raw training weights.

### Training Loop

```
for epoch in range(num_epochs):
    for batch in train_loader:
        t ~ importance_weighted_sample(T)
        ε ~ N(0, I)
        x_t = add_noise(x_0, ε, t)
        ε_θ = model(x_t, t)
        loss = MSE(ε_θ, ε) / accumulation_steps
        loss.backward()

        if step % accumulation_steps == 0:
            clip_grad_norm(1.0)
            optimizer.step()
            lr_scheduler.step()  # OneCycleLR
            ema.update(model)

    # Validate with EMA weights
    ema.apply(model)
    val_loss = evaluate(val_loader)
    ema.restore(model)

    # Early stopping on EMA val loss
    if no improvement for 25 epochs: break

# Load best EMA weights
model.load_state_dict(best_ema_shadow)
```

## Data Preprocessing

### QuantileTransformer

StandardScaler maps data to mean=0, std=1 — but heavy-tailed features (AveOccup, Population) get squashed to [-3, 3], losing tail information. The inverse transform can't recover what was lost.

QuantileTransformer maps each feature to a perfect Gaussian via rank ordering:
1. Compute empirical CDF (quantiles) for each feature
2. Map quantiles to the corresponding Gaussian values via `Φ⁻¹(quantile)`
3. The inverse transform reconstructs the original values faithfully from the quantile mapping

This is "lossless" normalization — the diffusion model learns in Gaussian space, and `inverse_transform` recovers the original distribution shape including tails.

## Evaluation

### Per-Feature KS Statistic

The Kolmogorov-Smirnov statistic measures the maximum difference between the empirical CDFs of real and synthetic data for each feature:

```
KS = max_x |F_real(x) - F_synthetic(x)|
```

- KS < 0.10: PASS (distributions nearly identical)
- KS < 0.20: WARN (distributions similar but not perfect)
- KS ≥ 0.20: FAIL (distributions diverge)

### Correlation Frobenius Norm

Measures how well the synthetic data preserves inter-feature correlations:

```
corr_diff = ||Corr(real) - Corr(synthetic)||_F
```

Lower is better. The Frobenius norm captures differences across all feature pairs.

### ML Utility Ratio

Trains a GradientBoostingRegressor on synthetic data and evaluates on held-out real data:

```
utility = R²(synthetic-trained) / R²(real-trained)
```

A ratio of 0.94 means a model trained entirely on synthetic data achieves 94% of the performance of one trained on real data.

### Composite Score

Weighted combination of all three metrics:

```
composite = w_ks · (1 - avg_ks) + w_corr · max(0, 1 - corr_diff) + w_ml · ml_utility
```

## Self-Improvement Loop

When `--self_improve` is enabled, the pipeline runs multiple iterations:

```
for iteration in range(max_iterations):
    1. Create model + scheduler
    2. Train
    3. Generate synthetic data
    4. Evaluate quality
    5. Check stopping conditions (threshold, convergence)
    6. Adjust hyperparameters based on diagnosis:
       - High KS → more epochs + lower LR
       - High corr diff → more capacity (wider, deeper)
       - Low ML utility → both
```

The loop stops when the composite score exceeds the quality threshold or converges (delta < 0.01 for 3 consecutive iterations).
