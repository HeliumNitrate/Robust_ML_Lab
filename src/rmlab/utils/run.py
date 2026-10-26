"""Run folders: one directory per run with everything needed to reproduce it."""

import json
import logging
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig

from rmlab import __version__
from rmlab.utils.config import save_config

REPO_ROOT = Path(__file__).resolve().parents[3]
log = logging.getLogger("rmlab")


def resolve_device(requested: str) -> str:
    """Turn ``auto`` / ``cpu`` / ``cuda`` into a device that really works.

    ``auto`` falls back to cpu with a warning. An explicit ``cuda`` that does not work
    raises instead of silently training on the cpu.
    """
    if requested not in ("auto", "cpu", "cuda"):
        raise ValueError(f"train.device must be auto, cpu or cuda, got {requested!r}")
    if requested == "cpu":
        return "cpu"
    if not torch.cuda.is_available():
        if requested == "cuda":
            raise RuntimeError("train.device=cuda, but CUDA is not available")
        return "cpu"
    try:
        torch.ones(1, device="cuda").add_(1)  # run a real kernel, not only a driver check
    except RuntimeError as err:
        if requested == "cuda":
            raise
        log.warning("CUDA is reported available but does not work (%s); using cpu", err)
        return "cpu"
    return "cuda"


def git_info(repo: Path = REPO_ROOT) -> dict[str, Any]:
    """Current commit and whether the working tree has uncommitted changes."""

    def git(*args: str) -> str:
        out = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=True)
        return out.stdout.strip()

    try:
        return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


class Run:
    """A run folder: resolved config, run info and log file."""

    def __init__(self, cfg: DictConfig) -> None:
        self.device = resolve_device(cfg.train.device)  # fail before creating anything

        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.dir = Path(cfg.paths.runs) / f"{stamp}_{cfg.name}_s{cfg.seed}"
        self.dir.mkdir(parents=True, exist_ok=False)  # never write into an old run
        self._log_handler = _setup_logging(self.dir / "train.log")

        save_config(cfg, self.dir / "config.yaml")
        self.info: dict[str, Any] = {
            "rmlab": __version__,
            "started": datetime.now().isoformat(timespec="seconds"),
            "command": sys.argv,
            "git": git_info(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "platform": platform.platform(),
            "resolved": {"device": self.device},
        }
        self._write_info()

        log.info("Run folder: %s", self.dir)
        if self.info["git"]["dirty"]:
            log.warning("Uncommitted changes: this run does not match a commit (dirty=true)")

    def record(self, **values: Any) -> None:
        """Add values derived at run time (e.g. number of classes) to run_info.json."""
        self.info["resolved"].update(values)
        self._write_info()

    def close(self) -> None:
        """Stop writing to this run's log file."""
        log.removeHandler(self._log_handler)
        self._log_handler.close()

    def _write_info(self) -> None:
        text = json.dumps(self.info, indent=2, default=str)
        (self.dir / "run_info.json").write_text(text, encoding="utf-8")


def _setup_logging(log_file: Path) -> logging.Handler:
    """Log to the console and to the run's log file."""
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
    if not any(type(h) is logging.StreamHandler for h in log.handlers):
        console = logging.StreamHandler()
        console.setFormatter(fmt)
        log.addHandler(console)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(fmt)
    log.addHandler(file_handler)
    return file_handler
