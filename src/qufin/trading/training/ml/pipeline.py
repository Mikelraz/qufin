"""
sklearn pipeline scaffolding with leak-safe walk-forward splits.

The splits here mirror ``walk_forward.WalkForwardCV`` but operate on a
feature matrix rather than a backtest engine — use these for model
selection / calibration when the loss function is a classification or
regression metric, then plug the resulting estimator into
``MLSignalStrategy`` for a proper PnL backtest.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np


def build_classifier_pipeline(
    estimator: Any | None = None,
    *,
    with_scaler: bool = True,
) -> Any:
    """sklearn ``Pipeline`` for classification.

    Default estimator is ``GradientBoostingClassifier``; supply your own to
    swap. ``with_scaler`` adds a ``StandardScaler`` step (no-op for trees
    but useful for linear/SVM heads).
    """
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    head = estimator if estimator is not None else GradientBoostingClassifier()
    steps: list[tuple[str, Any]] = []
    if with_scaler:
        steps.append(("scaler", StandardScaler()))
    steps.append(("estimator", head))
    return Pipeline(steps)


def build_regressor_pipeline(
    estimator: Any | None = None,
    *,
    with_scaler: bool = True,
) -> Any:
    """sklearn ``Pipeline`` for regression (e.g. return forecasting)."""
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    head = estimator if estimator is not None else GradientBoostingRegressor()
    steps: list[tuple[str, Any]] = []
    if with_scaler:
        steps.append(("scaler", StandardScaler()))
    steps.append(("estimator", head))
    return Pipeline(steps)


def walk_forward_splits(
    n: int,
    *,
    train_size: int,
    test_size: int,
    step: int | None = None,
    embargo: int = 0,
    expanding: bool = False,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield ``(train_idx, test_idx)`` numpy arrays for walk-forward CV.

    Identical splitting semantics as ``WalkForwardCV`` so model-selection
    results in feature space carry over to the trading engine without
    re-aligning indices.
    """
    step_ = step if step is not None else test_size
    train_start = 0
    cur_train_size = train_size
    while True:
        train_end = train_start + cur_train_size
        test_start = train_end + embargo
        test_end = test_start + test_size
        if test_end > n:
            return
        yield np.arange(train_start, train_end), np.arange(test_start, test_end)
        if expanding:
            cur_train_size += step_
        else:
            train_start += step_
