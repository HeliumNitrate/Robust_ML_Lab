"""Tests for loading, merging and saving experiment configs."""

import shutil
from pathlib import Path

import pytest
from omegaconf import OmegaConf
from omegaconf.errors import ConfigKeyError

from rmlab.utils.config import load_config, save_config

CONFIGS = Path(__file__).parents[1] / "configs"


@pytest.fixture
def config_dir(tmp_path: Path) -> Path:
    """A temporary configs folder holding a copy of the real base.yaml."""
    shutil.copy(CONFIGS / "base.yaml", tmp_path / "base.yaml")
    return tmp_path


def write_experiment(config_dir: Path, text: str) -> Path:
    path = config_dir / "exp.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_smoke_config_merges_base_and_experiment() -> None:
    cfg = load_config(CONFIGS / "smoke.yaml")
    assert cfg.name == "smoke"  # from smoke.yaml
    assert cfg.train.epochs == 1  # smoke.yaml overrides base
    assert cfg.train.optimizer.lr == 1e-3  # only in base


def test_cli_overrides_win() -> None:
    cfg = load_config(CONFIGS / "smoke.yaml", ["seed=3", "train.optimizer.lr=0.01"])
    assert cfg.seed == 3
    assert cfg.train.optimizer.lr == 0.01


def test_unknown_key_from_cli_raises() -> None:
    with pytest.raises(ConfigKeyError):
        load_config(CONFIGS / "smoke.yaml", ["train.epoch=5"])  # typo: epochs


def test_unknown_key_in_experiment_file_raises(config_dir: Path) -> None:
    path = write_experiment(
        config_dir, "name: x\ndata: {name: fake}\nmodel: {name: linear, chanels: 64}\n"
    )
    with pytest.raises(ConfigKeyError):
        load_config(path)


def test_params_accept_new_keys() -> None:
    cfg = load_config(CONFIGS / "smoke.yaml", ["model.params.channels=64"])
    assert cfg.model.params.channels == 64


def test_missing_required_value_raises(config_dir: Path) -> None:
    path = write_experiment(config_dir, "name: x\ndata: {name: fake}\n")  # no model.name
    with pytest.raises(ValueError, match="model.name"):
        load_config(path)


def test_paths_come_from_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("RMLAB_RUNS_DIR", str(tmp_path))
    cfg = load_config(CONFIGS / "smoke.yaml")
    assert cfg.paths.runs == str(tmp_path)


def test_save_config_writes_resolved_yaml(tmp_path: Path) -> None:
    cfg = load_config(CONFIGS / "smoke.yaml")
    out = tmp_path / "config.yaml"
    save_config(cfg, out)
    assert "${" not in out.read_text(encoding="utf-8")
    assert OmegaConf.load(out).train.epochs == 1
