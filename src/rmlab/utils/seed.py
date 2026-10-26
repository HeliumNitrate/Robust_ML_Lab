"""One source of randomness per experiment."""

import random

import numpy as np
import torch


def set_seed(seed: int) -> np.random.Generator:
    """Seed Python, NumPy and PyTorch from one integer.

    Returns a NumPy generator. Code that needs randomness (splits, sub-sampling,
    augmentation) should take this generator, or one derived from it, as an argument
    instead of creating its own with a hard-coded seed.
    """
    random.seed(seed)
    np.random.seed(seed)  # global state, for third-party code that uses it
    torch.manual_seed(seed)  # also seeds all CUDA devices
    return np.random.default_rng(seed)


def worker_init_fn(worker_id: int) -> None:
    """Give each DataLoader worker its own reproducible NumPy and Python seed.

    PyTorch already gives every worker a distinct torch seed derived from the loader's
    base seed. Without this function, NumPy and Python ``random`` would start from the
    same state in every worker, so random augmentations would repeat across workers.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
