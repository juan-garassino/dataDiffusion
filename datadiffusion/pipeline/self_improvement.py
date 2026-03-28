"""Self-improvement loop: Train → Synthesise → Compare → Diagnose → Adjust → Repeat."""

import os
import copy
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
import mlflow

from ..config import ExperimentConfig
from ..data import get_dataset
from ..data.loading import CALIFORNIA_FEATURE_NAMES
from ..models.factory import create_model_and_scheduler
from ..training.trainer import Trainer
from ..generation.sampler import generate_samples, generate_samples_ddim
from ..evaluation.composite import QualityMetrics, compute_quality_metrics
from ..evaluation.reports import print_quality_report, save_quality_report
from ..visualization.plots import plot_loss_curve
from ..visualization.comparison import plot_marginal_distributions, plot_correlation_comparison
from ..tracking.logger import get_logger
from ..tracking.mlflow_utils import setup_experiment, nested_run

logger = get_logger("datadiffusion.pipeline")


@dataclass
class IterationResult:
    iteration: int
    quality: QualityMetrics
    train_losses: List[float]
    val_losses: List[float]
    config_snapshot: dict


@dataclass
class LoopResult:
    iterations: List[IterationResult] = field(default_factory=list)
    best_iteration: int = 0

    @property
    def quality(self) -> QualityMetrics:
        return self.iterations[self.best_iteration].quality


class SelfImprovementLoop:
    """Orchestrates the full train → eval → adjust loop."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.si = config.self_improvement

    def run(self) -> LoopResult:
        dataset, transformer = get_dataset(
            self.config.dataset, scaler_type=self.config.data.scaler_type
        )
        real_X = dataset.tensors[0].numpy()
        real_y = dataset.tensors[1].numpy() if len(dataset.tensors) > 1 else None
        input_size = real_X.shape[1]

        feature_names = CALIFORNIA_FEATURE_NAMES if self.config.dataset == "california" else None

        max_iter = self.si.max_iterations if self.si.enabled else 1
        outdir = setup_experiment(self.config.experiment_name)

        result = LoopResult()
        best_score = -1.0
        stale_count = 0
        prev_score = -1.0

        for iteration in range(max_iter):
            iter_dir = os.path.join(outdir, f"iter_{iteration}")
            os.makedirs(iter_dir, exist_ok=True)

            logger.info("=== Iteration %d ===", iteration)

            with nested_run(f"iter_{iteration}"):
                # 1. Create model + scheduler
                model, scheduler = create_model_and_scheduler(self.config, input_size)

                # 2. Train
                trainer = Trainer(self.config)
                train_losses, val_losses = trainer.train(model, scheduler, dataset)

                # 3. Generate
                device = next(model.parameters()).device
                gen_cfg = self.config.generation
                if gen_cfg.sampler_type == "ddim" and self.config.scheduler.scheduler_type == "standard":
                    synthetic = generate_samples_ddim(
                        model, scheduler,
                        num_samples=len(real_X),
                        input_size=input_size,
                        num_inference_steps=gen_cfg.ddim_steps,
                        eta=0.0,
                    )
                else:
                    synthetic = generate_samples(
                        model, scheduler,
                        num_samples=len(real_X),
                        input_size=input_size,
                    )

                # 4. Inverse-transform to original feature space for evaluation
                real_X_orig = transformer.inverse_transform(real_X)
                synthetic_orig = transformer.inverse_transform(synthetic)

                quality = compute_quality_metrics(real_X_orig, synthetic_orig, real_y, feature_names)

                # 5. Report
                print_quality_report(quality, iteration, self.si.quality_threshold)
                save_quality_report(quality, iter_dir, iteration, self.si.quality_threshold)

                # 6. Save artifacts
                plot_loss_curve(train_losses, val_losses, os.path.join(iter_dir, "loss_curve.png"))
                plot_marginal_distributions(real_X_orig, synthetic_orig, feature_names, os.path.join(iter_dir, "marginal_histograms.png"))
                plot_correlation_comparison(real_X_orig, synthetic_orig, os.path.join(iter_dir, "correlation_comparison.png"))
                torch.save(model.state_dict(), os.path.join(iter_dir, "model_checkpoint.pt"))

                # 7. Log to MLflow
                mlflow.log_metrics({
                    "composite_score": quality.composite_score,
                    "avg_ks": quality.avg_ks_statistic,
                    "corr_diff_norm": quality.corr_diff_norm,
                    "ml_utility_ratio": quality.ml_utility_ratio,
                    "train_loss_final": train_losses[-1] if train_losses else 0,
                    "val_loss_final": val_losses[-1] if val_losses else 0,
                })
                for artifact in ["loss_curve.png", "marginal_histograms.png", "correlation_comparison.png", "quality_report.txt"]:
                    path = os.path.join(iter_dir, artifact)
                    if os.path.exists(path):
                        mlflow.log_artifact(path)

            iter_result = IterationResult(
                iteration=iteration,
                quality=quality,
                train_losses=train_losses,
                val_losses=val_losses,
                config_snapshot=self.config.to_dict(),
            )
            result.iterations.append(iter_result)

            # Track best
            if quality.composite_score > best_score:
                best_score = quality.composite_score
                result.best_iteration = iteration

            # 8. Check stopping conditions
            if quality.composite_score >= self.si.quality_threshold:
                logger.info("Quality threshold reached (%.3f >= %.3f) — stopping.",
                            quality.composite_score, self.si.quality_threshold)
                break

            if not self.si.enabled:
                break

            # Convergence check
            delta = quality.composite_score - prev_score
            if abs(delta) < self.si.convergence_delta:
                stale_count += 1
                if stale_count >= 3:
                    logger.info("Converged (delta < %.4f for 3 iterations) — stopping.", self.si.convergence_delta)
                    break
            else:
                stale_count = 0
            prev_score = quality.composite_score

            # 9. Adjust hyperparameters
            self._adjust_config(quality)

        # Print summary
        logger.info("Loop complete. Best score: %.4f at iteration %d", best_score, result.best_iteration)
        self._print_summary(result)

        mlflow.end_run()
        return result

    def _adjust_config(self, quality: QualityMetrics):
        """Heuristic hyperparameter adjustment based on diagnosis."""
        mc = self.config.model
        tc = self.config.training
        si = self.si

        # Bad per-feature KS → more epochs + lower LR
        if quality.avg_ks_statistic > 0.3:
            tc.num_epochs = int(tc.num_epochs * si.epoch_increase_factor)
            tc.learning_rate *= si.lr_decay_factor
            logger.info("Adjusting: epochs→%d, lr→%.2e (high KS)", tc.num_epochs, tc.learning_rate)

        # Bad correlation → more capacity
        if quality.corr_diff_norm > 0.5:
            mc.hidden_size = int(mc.hidden_size * si.capacity_increase_factor)
            mc.num_layers = mc.num_layers + 1
            logger.info("Adjusting: hidden_size→%d, num_layers→%d (high corr diff)", mc.hidden_size, mc.num_layers)

        # Bad ML utility → both
        if quality.ml_utility_ratio < 0.5:
            tc.num_epochs = int(tc.num_epochs * si.epoch_increase_factor)
            mc.hidden_size = int(mc.hidden_size * si.capacity_increase_factor)
            logger.info("Adjusting: epochs→%d, hidden_size→%d (low ML utility)", tc.num_epochs, mc.hidden_size)

    def _print_summary(self, result: LoopResult):
        print("\n" + "=" * 60)
        print("SELF-IMPROVEMENT SUMMARY")
        print("=" * 60)
        for ir in result.iterations:
            marker = " ← best" if ir.iteration == result.best_iteration else ""
            print(f"  Iter {ir.iteration}: composite={ir.quality.composite_score:.4f}  "
                  f"KS={ir.quality.avg_ks_statistic:.3f}  "
                  f"corr={ir.quality.corr_diff_norm:.3f}  "
                  f"ml_util={ir.quality.ml_utility_ratio:.3f}{marker}")
        print("=" * 60)
