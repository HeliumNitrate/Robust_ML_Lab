"""Tests for seeding."""

import random

import numpy as np
import torch

from rmlab.utils.seed import set_seed, worker_init_fn


def draw(seed: int) -> tuple[float, float, float, float]:
    rng = set_seed(seed)
    return random.random(), float(np.random.rand()), float(torch.rand(1)), float(rng.random())


def test_same_seed_gives_same_numbers() -> None:
    assert draw(7) == draw(7)


def test_different_seeds_give_different_numbers() -> None:
    assert draw(7) != draw(8)


def test_worker_init_fn_follows_torch_seed() -> None:
    def worker_draw(torch_seed: int) -> float:
        torch.manual_seed(torch_seed)  # DataLoader does this per worker
        worker_init_fn(0)
        return float(np.random.rand())

    assert worker_draw(100) == worker_draw(100)
    assert worker_draw(100) != worker_draw(101)
