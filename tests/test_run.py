"""Tests for run folders and device resolution."""

import json
import re
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from rmlab.utils.config import load_config
from rmlab.utils.run import Run, resolve_device

SMOKE = Path(__file__).parents[1] / "configs" / "smoke.yaml"


@pytest.fixture
def run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("RMLAB_RUNS_DIR", str(tmp_path))
    r = Run(load_config(SMOKE, ["seed=3", "train.device=cpu"]))
    yield r
    r.close()  # release train.log so the temporary folder can be removed (Windows)


def test_run_folder_name_and_location(run: Run, tmp_path: Path) -> None:
    assert run.dir.parent == tmp_path
    assert re.fullmatch(r"\d{8}-\d{6}_smoke_s3", run.dir.name)


def test_run_folder_contents(run: Run) -> None:
    assert {p.name for p in run.dir.iterdir()} == {"config.yaml", "run_info.json", "train.log"}
    assert OmegaConf.load(run.dir / "config.yaml").seed == 3

    info = json.loads((run.dir / "run_info.json").read_text(encoding="utf-8"))
    assert set(info["git"]) == {"commit", "dirty"}
    assert info["resolved"] == {"device": "cpu"}


def test_record_adds_resolved_values(run: Run) -> None:
    run.record(num_classes=4)
    info = json.loads((run.dir / "run_info.json").read_text(encoding="utf-8"))
    assert info["resolved"] == {"device": "cpu", "num_classes": 4}


def test_resolve_device() -> None:
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("auto") in {"cpu", "cuda"}
    with pytest.raises(ValueError):
        resolve_device("gpu")


@pytest.mark.skipif(torch.cuda.is_available(), reason="needs a machine without CUDA")
def test_explicit_cuda_without_cuda_raises() -> None:
    with pytest.raises(RuntimeError):
        resolve_device("cuda")
