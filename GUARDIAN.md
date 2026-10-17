# GUARDIAN

Automated security & bug-hygiene ledger for `data-diffusion`. Updated on every Guardian run
on branch `claude/guardian`.

## Checks

| Area | Tool / rule |
|---|---|
| Dependencies | `pip-audit -r requirements.txt` |
| Static security | `bandit -r datadiffusion main.py` |
| Risky patterns (grep) | eval/exec, pickle / `torch.load` without `weights_only=True`, `yaml.load` without SafeLoader, `subprocess(shell=True)`, SQL string formatting, path traversal, SSRF, `verify=False`, permissive CORS, debug servers on 0.0.0.0 |
| Secrets | `gitleaks detect --log-opts=--all --redact` (full history), `detect-secrets scan --all-files` |
| CI / containers | GitHub Actions workflows, Dockerfiles (none present in repo) |
| Tests | `pytest tests/` in a venv with the PyPI `torch` wheel (CPU use works; CUDA runtime packages get installed too) |
| Lint (metrics only) | `ruff check .` (ruff defaults; repo has no linter config) |
| Code review | manual read of CLI, config, trainer, losses, schedulers, sampler, pipeline |

## Findings ledger

| id | severity | type | file | status | date |
|---|---|---|---|---|---|
| G-001 | medium | bug (crash) | `datadiffusion/training/trainer.py` | fixed | 2026-10-15 |
| G-002 | low | bug (crash) | `datadiffusion/cli.py` | fixed | 2026-10-15 |
| G-003 | low | bug (silent misconfig) | `datadiffusion/cli.py` | fixed | 2026-10-15 |
| G-004 | low | info disclosure (local paths / username in tracked `mlruns/`) | `mlruns/` | open | 2026-10-15 |
| G-005 | low | bandit B105 false positive (`'✓ PASS'` label) | `datadiffusion/evaluation/reports.py` | wontfix | 2026-10-15 |
| G-006 | low | secrets-scan false positive (MLflow run ids flagged as hex entropy) | `mlruns/` | wontfix | 2026-10-15 |
| G-007 | low | quality (unused local `device`) | `datadiffusion/pipeline/self_improvement.py` | fixed | 2026-10-15 |
| G-008 | medium | bug (crash) | `datadiffusion/pipeline/self_improvement.py` | fixed | 2026-10-17 |
| G-009 | low | bug (silent wrong output / crash) | `datadiffusion/generation/sampler.py` | fixed | 2026-10-17 |
| G-010 | medium | bug (silent wrong sampling) | `datadiffusion/models/factory.py`, `README.md` | mitigated (warning) | 2026-10-17 |
| G-011 | medium | bug (LR schedule never anneals) | `datadiffusion/training/trainer.py` | fixed | 2026-10-17 |

Details:

- **G-001**: `--model_type score` with the default `--scheduler_type standard` raised
  `AttributeError` at the first training step: the trainer read
  `scheduler.sqrt_1m_alphas_cumprod`, which only exists on `ScoreBasedNoiseScheduler`. The trainer
  now reads `sqrt_one_minus_alphas_cumprod`, which both schedulers define with identical values.
  Test: `tests/test_training.py::test_score_model_with_standard_scheduler_no_crash`.
- **G-002**: the CLI listed `--dataset moons|line|circle`, but `get_dataset` only implements
  `california`, so those choices crashed after setup. The choices are now limited to `california`
  and the README is updated. Test: `tests/test_cli.py::test_unsupported_dataset_rejected_at_parse_time`.
- **G-003**: `--config <path>` was silently ignored when the file did not exist, so a typo ran a full
  experiment with default settings. It is now an argparse error. Test:
  `tests/test_cli.py::test_missing_config_file_is_an_error`.
- **G-004**: `mlruns/` is in `.gitignore` but its files are still tracked. Their `meta.yaml` files
  reveal a local absolute path and username. Suggest `git rm -r --cached mlruns` (owner decision;
  the change would touch about 100 files).
- **G-008**: in `--self_improve`, `capacity_increase_factor=1.5` grew `hidden_size` to values not
  divisible by `num_heads` (256 → 384 → 576 → … → 4374 with 4 heads). The next transformer or score
  model then crashed when it was built. `hidden_size` is now rounded up to a multiple of
  `lcm(num_heads, 2)`. Test: `tests/test_pipeline.py::test_adjust_config_keeps_hidden_size_divisible_by_num_heads`.
- **G-009**: `generate_samples_ddim` with `num_inference_steps > num_timesteps` gave `step_ratio = 0`,
  so every DDIM step used t=0 and the output was silently wrong. A value of 0 raised
  `ZeroDivisionError`. Values outside `[1, num_timesteps]` now raise a `ValueError`. Test:
  `tests/test_sampler.py::test_ddim_rejects_invalid_num_inference_steps` (0, -1, 101).
- **G-010**: score models output a score, while mlp and transformer models output noise. The
  scheduler's reverse step reads the model output as one or the other. So `score` + `standard`
  and `mlp`/`transformer` + `score` sample with the wrong sign and scale. The README even showed
  `--model_type transformer --scheduler_type score` as an example. `create_model_and_scheduler` now
  logs a "model/scheduler mismatch" warning, and the README example now uses `score` + `score`.
  It is a warning, not an error, because the trainer accepts both combinations (see the G-001
  test). Making it a hard error is left to the owner. Test:
  `tests/test_models.py::test_factory_warns_on_mismatched_model_scheduler` (5 cases).
- **G-011**: `OneCycleLR` was sized with `steps_per_epoch=len(train_loader)`, but the optimizer
  only steps once per `accumulation_steps` window (default 4). Training therefore used only about
  1/4 of the schedule. The LR never reached its peak (`pct_start=0.3`) and never annealed. The
  schedule is now sized as `ceil(len(train_loader) / accumulation_steps)`. Test:
  `tests/test_training.py::test_lr_schedule_spans_all_optimizer_steps` (failed before the fix
  with 4 of 16 steps used). Training results from before this fix (for example `results/`) were
  produced with this schedule.

## Metrics

| Metric | Run 1 before | Run 1 after | Run 2 after |
|---|---|---|---|
| Open vulns (crit/high/med/low) | 0/0/0/1 (G-004) | 0/0/0/1 (G-004) | 0/0/0/1 (G-004) |
| Open bugs found in run | 3 | 0 | 0 (4 found and fixed; G-010 mitigated) |
| Dependency CVEs (pip-audit) | 0 | 0 | 0 |
| Secrets (gitleaks, full history) | 0 | 0 | 0 |
| Bandit findings | 1 (low, FP) | 1 | 1 |
| Test cases collected | not measurable | not measurable | 48 before → 58 after |
| Passing tests (full suite) | not measurable | CLI 6/6 | 43/48 before → 53/58 after (all 5 failures need the California Housing download, which the sandbox proxy blocks) |
| Ruff findings (defaults) | 101 | 100 | 100 |
| Coverage | not measurable | not measurable | not measured |

## Backlog

- Untrack `mlruns/` (G-004).
- Make the California Housing tests run offline: cache the dataset or add a fixture that skips when
  the download fails. They are the only 5 failing tests in the sandbox.
- G-010 owner decision: turn the model/scheduler mismatch warning into a hard error in
  `ExperimentConfig` or the CLI.
- `ScoreBasedNoiseScheduler.step` multiplies the per-step discrete `betas` by `dt = 1/T` again.
  The reverse SDE therefore barely moves the sample in total. This needs a numerical check against
  the forward marginals, so the rest of the score/SDE path should be reviewed first.
- `_sample_timesteps` importance sampling (weights 1/(t+1)) is heavily biased toward t≈0 and the
  loss is not reweighted. Check whether this is intentional.
- Measure coverage (`pytest --cov`) now that torch can be installed.
- Add a `pyproject.toml` / ruff config. About half of the 100 lint findings can be fixed automatically
  (unused imports and import order).

## Log

- **Run 1 (2026-10-15):** initial scan. No dependency CVEs and no secrets in the full history. Bandit
  reported only one low-severity false positive. The repo has no CI workflows or Dockerfiles. Fixed 3
  crash or misconfiguration bugs (G-001..G-003), each with a regression test written first, plus 1
  lint item. The torch test suites could not run in the sandbox (no CPU torch wheel was reachable).
  The CLI tests ran and were confirmed failing before the fix and passing after.

- **Run 2 (2026-10-17):** no review comments on the PR. Merged `origin/master`, which was already
  up to date. The `.git/hooks/pre-push` install was denied by the sandbox permission classifier, so
  the run pushed only an explicit `claude/guardian` refspec. Rescan results: pip-audit 0, bandit 1
  (known FP), gitleaks 0 across 25 commits, still no CI workflows or Dockerfiles. Got the full torch
  suite running for the first time: 43/48 before, 53/58 after. Cleared 3 backlog items and found 1
  new bug by reading code (G-011). For each one, a failing test was confirmed before the fix. Ruff
  stayed at 100.

## Learnings

- In the Guardian sandbox, the PyPI `torch` wheel needs the CUDA runtime packages (several GB), and
  the CPU wheel index is not reachable. Run torch-dependent tests elsewhere or with a cached CPU wheel.
- `tests/conftest.py` imports torch, so non-torch tests need `--noconftest` to run without it.
- (Run 2) The full PyPI `torch` wheel installs fast through the proxy, and CPU use works once its
  CUDA runtime packages are installed. Installing with `--no-deps` fails at import
  (`libcublasLt` not found). The CPU wheel index is still unreachable.
- (Run 2) sklearn's `fetch_california_housing` source (figshare) is blocked by the proxy (403).
  Tests that need it fail for environmental reasons, not because of code.
- (Run 2) Set `MLFLOW_TRACKING_URI` to a scratch directory when you run tests, so they do not write
  into the tracked `mlruns/`.

## Ideas

1. Add a minimal GitHub Actions workflow (pinned SHAs, `permissions: contents: read`) that runs
   `pytest -m "not slow"` on CPU torch.
2. Validate `model_type` / `scheduler_type` / `sampler_type` combinations in `ExperimentConfig.__post_init__`.
3. Run `ruff --fix` for imports only, as a separate low-risk PR.
4. Add a smoke test that runs one optimizer step per (model_type, scheduler_type, sampler_type)
   combination and asserts that the samples are finite and not exploding.
5. Add a `pytest.ini` marker (`network`) for tests that download data, so the offline CI can
   deselect them.
6. Add a regression test that checks the EMA shadow is only updated on optimizer steps (it
   currently is). It would guard future changes to the accumulation logic.
