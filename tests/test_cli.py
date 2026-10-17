"""Tests for CLI argument parsing (Guardian run 1)."""

import pytest

from datadiffusion.cli import parse_args


def test_default_dataset_is_california():
    assert parse_args([]).dataset == "california"


@pytest.mark.parametrize("name", ["moons", "line", "circle"])
def test_unsupported_dataset_rejected_at_parse_time(name):
    # get_dataset() only implements "california"; advertising other choices
    # let the CLI accept them and then crash deep inside the pipeline.
    with pytest.raises(SystemExit):
        parse_args(["--dataset", name])


def test_missing_config_file_is_an_error(tmp_path):
    # A typo in --config used to be silently ignored, running with defaults.
    with pytest.raises(SystemExit):
        parse_args(["--config", str(tmp_path / "does_not_exist.yaml")])


def test_existing_config_file_is_loaded(tmp_path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("experiment_name: from_yaml\ntraining:\n  num_epochs: 3\n")
    config = parse_args(["--config", str(cfg)])
    assert config.experiment_name == "from_yaml"
    assert config.training.num_epochs == 3
