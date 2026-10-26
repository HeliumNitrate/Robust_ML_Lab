"""Tests for name -> builder registries."""

import pytest

from rmlab.registry import Registry


def test_register_and_build() -> None:
    reg = Registry("thing")

    @reg.register("double")
    def double(x: int) -> int:
        return 2 * x

    assert reg.build("double", x=3) == 6
    assert reg.names() == ["double"]


def test_duplicate_name_raises() -> None:
    reg = Registry("thing")
    reg.register("a")(lambda: 1)
    with pytest.raises(ValueError):
        reg.register("a")(lambda: 2)


def test_unknown_name_lists_available() -> None:
    reg = Registry("thing")
    reg.register("a")(lambda: 1)
    with pytest.raises(KeyError, match="available: \\['a'\\]"):
        reg.build("b")


def test_unknown_argument_raises() -> None:
    reg = Registry("thing")
    reg.register("a")(lambda size: size)
    with pytest.raises(TypeError):
        reg.build("a", sise=3)  # typo in a params key is caught here
