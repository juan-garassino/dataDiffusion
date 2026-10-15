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
| Tests | `pytest tests/` |
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

## Metrics

| Metric | Before | After |
|---|---|---|
| Open vulns (crit/high/med/low) | 0/0/0/1 (G-004) | 0/0/0/1 (G-004) |
| Open bugs found this run | 3 | 0 |
| Dependency CVEs (pip-audit) | 0 | 0 |
| Secrets (gitleaks, full history) | 0 | 0 |
| Test functions | 33 | 38 (+4 CLI incl. 1 parametrized x3, +1 training) |
| Failing tests | not measurable (see Learnings) | CLI suite 6/6 pass; torch suites not run |
| Ruff findings (defaults) | 101 | 100 |
| Coverage | not measurable | not measurable |

## Backlog

- Untrack `mlruns/` (G-004).
- Run the full torch test suite in an environment that can install CPU torch, and record test results and coverage.
- `TabularTransformer` with `--self_improve`: `capacity_increase_factor=1.5` eventually produces a
  `hidden_size` that is not divisible by `num_heads` (for example 2916 → 4374 with 4 heads). Clamp it
  to a multiple of `num_heads`.
- `generate_samples_ddim`: `num_inference_steps > num_timesteps` gives `step_ratio = 0`, so every
  step uses t=0. Validate the input or clamp it.
- `--model_type score` with `scheduler_type standard` still samples through the DDPM `step`, which
  treats the predicted score as noise. Consider requiring `scheduler_type=score` for score models.
- Add a `pyproject.toml` / ruff config. About half of the 100 lint findings can be fixed automatically
  (unused imports and import order).

## Log

- **Run 1 (2026-10-15):** initial scan. No dependency CVEs and no secrets in the full history. Bandit
  reported only one low-severity false positive. The repo has no CI workflows or Dockerfiles. Fixed 3
  crash or misconfiguration bugs (G-001..G-003), each with a regression test written first, plus 1
  lint item. The torch test suites could not run in the sandbox (no CPU torch wheel was reachable).
  The CLI tests ran and were confirmed failing before the fix and passing after.

## Learnings

- In the Guardian sandbox, the PyPI `torch` wheel needs the CUDA runtime packages (several GB), and
  the CPU wheel index is not reachable. Run torch-dependent tests elsewhere or with a cached CPU wheel.
- `tests/conftest.py` imports torch, so non-torch tests need `--noconftest` to run without it.

## Ideas

1. Add a minimal GitHub Actions workflow (pinned SHAs, `permissions: contents: read`) that runs
   `pytest -m "not slow"` on CPU torch.
2. Validate `model_type` / `scheduler_type` / `sampler_type` combinations in `ExperimentConfig.__post_init__`.
3. Run `ruff --fix` for imports only, as a separate low-risk PR.
