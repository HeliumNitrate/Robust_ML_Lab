"""Smoke tests: the package and its subpackages import cleanly."""

import importlib

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
