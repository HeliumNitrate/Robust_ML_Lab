"""Experiment configs: load, merge and save YAML files with OmegaConf."""

from pathlib import Path

from omegaconf import DictConfig, OmegaConf

BASE_FILE = "base.yaml"


def load_config(path: str | Path, overrides: list[str] | None = None) -> DictConfig:
    """Load an experiment config on top of the defaults.

    Merge order, later wins: ``base.yaml`` next to the experiment file, the experiment
    file, then command-line overrides such as ``["seed=1", "train.epochs=5"]``.

    The result is strict: a key that is not in ``base.yaml`` raises an error, so typos
    are caught. The ``params`` sections are the exception; their contents are passed to
    component builders, which reject unknown arguments themselves.
    """
    path = Path(path)
    base = OmegaConf.load(path.parent / BASE_FILE)
    OmegaConf.set_struct(base, True)
    for key in base:
        if OmegaConf.is_missing(base, key):
            continue
        section = base[key]
        if isinstance(section, DictConfig) and "params" in section:
            OmegaConf.set_struct(section.params, False)

    experiment = OmegaConf.load(path)
    cli = OmegaConf.from_dotlist(overrides or [])
    cfg = OmegaConf.merge(base, experiment, cli)

    missing = sorted(OmegaConf.missing_keys(cfg))
    if missing:
        raise ValueError(f"{path}: required config values are not set: {missing}")
    return cfg


def save_config(cfg: DictConfig, path: str | Path) -> None:
    """Write the config to a YAML file with all ``${...}`` references resolved."""
    Path(path).write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")
