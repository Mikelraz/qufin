"""Classical-ML training utilities (sklearn-backed)."""

from __future__ import annotations

from .features import (
    FeatureSet,
    build_default_features,
    rsi_feature,
    sma_ratio_feature,
)
from .pipeline import build_classifier_pipeline, build_regressor_pipeline, walk_forward_splits
from .signal import MLSignalStrategy

__all__ = [
    "FeatureSet",
    "MLSignalStrategy",
    "build_classifier_pipeline",
    "build_default_features",
    "build_regressor_pipeline",
    "rsi_feature",
    "sma_ratio_feature",
    "walk_forward_splits",
]
