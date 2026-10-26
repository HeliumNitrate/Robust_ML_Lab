"""Common data types: one Sample, and a torch Dataset over a list of samples."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

SPLITS = ("train", "val", "test")


@dataclass
class Sample:
    """One example. ``meta`` carries extra information such as ``source_id`` or SNR."""

    x: np.ndarray
    y: int | None
    meta: dict[str, Any] = field(default_factory=dict)


class SampleDataset(Dataset):
    """Turns a sequence of samples into ``(x, y)`` float32 / int64 tensors, one at a time."""

    def __init__(self, samples: Sequence[Sample]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        s = self.samples[index]
        return torch.as_tensor(s.x, dtype=torch.float32), torch.tensor(s.y, dtype=torch.int64)
