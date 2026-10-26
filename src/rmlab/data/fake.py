"""Fake dataset: Gaussian blobs, for smoke tests of the training pipeline."""

import numpy as np

from rmlab.data.base import SPLITS, Sample
from rmlab.registry import DATASETS


@DATASETS.register("fake")
class FakeDataset:
    """``num_classes`` Gaussian blobs in ``n_features`` dimensions.

    The class centres depend only on ``seed``, so all splits share one distribution;
    each split draws its own samples. ``seed`` is part of the dataset's identity, like a
    file name, and is independent of the run seed that drives model initialisation.
    """

    def __init__(
        self,
        num_classes: int = 3,
        n_features: int = 8,
        n_per_class: int = 100,
        separation: float = 3.0,
        seed: int = 0,
    ) -> None:
        self.num_classes = num_classes
        self.n_features = n_features
        self.n_per_class = n_per_class
        self.seed = seed
        rng = np.random.default_rng(seed)
        self.centres = separation * rng.standard_normal((num_classes, n_features))

    def load(self, split: str) -> list[Sample]:
        """Samples of one split, in class order (shuffling is the DataLoader's job)."""
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
        rng = np.random.default_rng([self.seed, SPLITS.index(split) + 1])
        samples = []
        for label, centre in enumerate(self.centres):
            noise = rng.standard_normal((self.n_per_class, self.n_features))
            for i, x in enumerate((centre + noise).astype(np.float32)):
                samples.append(Sample(x, label, {"source_id": f"fake-{split}-{label}-{i}"}))
        return samples
