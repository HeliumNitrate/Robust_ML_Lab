"""Tests for the fake dataset."""

import numpy as np
import pytest

from rmlab.data.base import SampleDataset
from rmlab.data.fake import FakeDataset


def test_shapes_and_labels() -> None:
    samples = FakeDataset(num_classes=4, n_features=5, n_per_class=10).load("train")
    assert len(samples) == 40
    assert samples[0].x.shape == (5,) and samples[0].x.dtype == np.float32
    assert {s.y for s in samples} == {0, 1, 2, 3}


def test_same_seed_same_data_and_splits_differ() -> None:
    a = FakeDataset(seed=1).load("train")
    b = FakeDataset(seed=1).load("train")
    val = FakeDataset(seed=1).load("val")
    assert np.array_equal(a[0].x, b[0].x)
    assert not np.array_equal(a[0].x, val[0].x)


def test_source_ids_do_not_repeat_across_splits() -> None:
    data = FakeDataset(n_per_class=5)
    ids = [s.meta["source_id"] for split in ("train", "val", "test") for s in data.load(split)]
    assert len(ids) == len(set(ids))


def test_unknown_split_raises() -> None:
    with pytest.raises(ValueError):
        FakeDataset().load("validation")


def test_sample_dataset_returns_tensors() -> None:
    x, y = SampleDataset(FakeDataset(n_features=6).load("val"))[0]
    assert x.shape == (6,) and str(x.dtype) == "torch.float32" and y.item() == 0
