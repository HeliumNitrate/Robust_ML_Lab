"""Name -> builder registries, so configs can refer to components by name."""

from collections.abc import Callable
from typing import Any


class Registry:
    """Maps names such as ``"linear"`` to the function or class that builds them."""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._builders: dict[str, Callable[..., Any]] = {}

    def register(self, name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator: ``@MODELS.register("linear")`` above a builder function or class."""

        def add(builder: Callable[..., Any]) -> Callable[..., Any]:
            if name in self._builders:
                raise ValueError(f"{self.kind} {name!r} is already registered")
            self._builders[name] = builder
            return builder

        return add

    def build(self, name: str, **kwargs: Any) -> Any:
        """Call the builder registered under ``name``; unknown keyword arguments raise."""
        if name not in self._builders:
            raise KeyError(f"unknown {self.kind} {name!r}; available: {self.names()}")
        return self._builders[name](**kwargs)

    def names(self) -> list[str]:
        return sorted(self._builders)


DATASETS = Registry("dataset")
MODELS = Registry("model")
