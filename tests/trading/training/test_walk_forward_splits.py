"""Walk-forward split semantics — no leakage and exhaustive coverage."""

from __future__ import annotations

import numpy as np

from qufin.trading.training.ml.pipeline import walk_forward_splits


def test_splits_no_overlap_no_embargo():
    folds = list(walk_forward_splits(n=100, train_size=40, test_size=20))
    assert len(folds) == 3  # 0-40/40-60, 20-60/60-80, 40-80/80-100
    for train_idx, test_idx in folds:
        assert train_idx.max() < test_idx.min()


def test_embargo_drops_buffer_between_train_and_test():
    folds = list(walk_forward_splits(n=100, train_size=40, test_size=20, embargo=5))
    for train_idx, test_idx in folds:
        # First test index must be at least train.end + embargo.
        assert test_idx.min() == train_idx.max() + 1 + 5


def test_expanding_grows_train_window():
    folds = list(walk_forward_splits(n=100, train_size=40, test_size=20, expanding=True))
    train_sizes = [len(t) for t, _ in folds]
    assert train_sizes == sorted(train_sizes)  # monotonically non-decreasing
    assert train_sizes[-1] > train_sizes[0]


def test_no_split_when_window_too_large():
    folds = list(walk_forward_splits(n=30, train_size=40, test_size=20))
    assert folds == []


def test_splits_are_contiguous_integer_ranges():
    for train_idx, test_idx in walk_forward_splits(n=100, train_size=40, test_size=20):
        assert np.all(np.diff(train_idx) == 1)
        assert np.all(np.diff(test_idx) == 1)
