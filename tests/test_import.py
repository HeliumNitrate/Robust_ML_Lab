"""Smoke tests: the package and its subpackages import cleanly."""

import importlib
from pathlib import Path

import pytest

import rmlab

SUBPACKAGES = ["data", "models", "train", "eval", "scoring", "augment", "explain", "utils"]


def test_version_is_set() -> None:
    assert isinstance(rmlab.__version__, str)
    assert rmlab.__version__ != ""


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_subpackage_imports(name: str) -> None:
    module = importlib.import_module(f"rmlab.{name}")
    assert module.__name__ == f"rmlab.{name}"


def test_package_is_imported_from_this_repo() -> None:
    """Fails if rmlab is a copied (non-editable) install: then edits and pulls are ignored."""
    src = Path(__file__).resolve().parents[1] / "src"
    assert Path(rmlab.__file__).resolve().is_relative_to(src), (
        f'rmlab is loaded from {rmlab.__file__}; reinstall with: python -m pip install -e ".[dev]"'
    )
