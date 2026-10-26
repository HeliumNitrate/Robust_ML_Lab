"""End-to-end test of the training command on fake data."""

import json
from pathlib import Path

import pytest

from rmlab.train.__main__ import main

SMOKE = str(Path(__file__).parents[1] / "configs" / "smoke.yaml")


def read_metrics(run_dir: Path) -> dict:
    return json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))


def test_smoke_run_writes_outputs_and_learns(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("RMLAB_RUNS_DIR", str(tmp_path))
    run = main(["--config", SMOKE, "train.epochs=3"])
    files = {p.name for p in run.dir.iterdir()}
    assert files == {"config.yaml", "run_info.json", "train.log", "metrics.json", "model.pt"}

    metrics = read_metrics(run.dir)
    assert len(metrics["history"]) == 3
    assert metrics["final"]["val_accuracy"] > 0.8  # well separated blobs

    info = json.loads((run.dir / "run_info.json").read_text(encoding="utf-8"))
    assert info["resolved"]["num_classes"] == 3


def test_same_seed_gives_same_metrics(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("RMLAB_RUNS_DIR", str(tmp_path))
    first = main(["--config", SMOKE, "train.epochs=2", "name=a"])
    second = main(["--config", SMOKE, "train.epochs=2", "name=b"])
    assert read_metrics(first.dir) == read_metrics(second.dir)
