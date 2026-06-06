"""
Shape statistics of a volume-at-price distribution.

These operate on a :class:`~qufin.volume_distribution._types.VolumeProfile`'s
``volume`` histogram (over bin centres):

* ``volume_concentration`` — Gini coefficient (0 = uniform, →1 = single bin).
* ``volume_entropy``       — normalised Shannon entropy (1 = uniform).
* ``profile_skew`` / ``profile_kurtosis`` — volume-weighted moments of price.
* ``classify_profile_shape`` — coarse normal / b / p / D label plus the above.
"""

from __future__ import annotations

import numpy as np

from ._types import DistributionStats, ProfileShape, VolumeProfile

__all__ = [
    "classify_profile_shape",
    "profile_kurtosis",
    "profile_skew",
    "volume_concentration",
    "volume_entropy",
]


def volume_concentration(profile: VolumeProfile) -> float:
    """Gini coefficient of the volume histogram in ``[0, 1]``."""
    v = np.sort(profile.volume.astype(np.float64, copy=False))
    n = v.shape[0]
    total = v.sum()
    if n == 0 or total <= 0.0:
        return 0.0
    ranks = np.arange(1, n + 1, dtype=np.float64)
    return float((2.0 * np.sum(ranks * v) / (n * total)) - (n + 1.0) / n)


def volume_entropy(profile: VolumeProfile) -> float:
    """Normalised Shannon entropy of the volume histogram in ``[0, 1]``."""
    v = profile.volume.astype(np.float64, copy=False)
    n = v.shape[0]
    total = v.sum()
    if n <= 1 or total <= 0.0:
        return 0.0
    p = v[v > 0.0] / total
    h = -np.sum(p * np.log(p))
    return float(h / np.log(n))


def _weighted_moments(profile: VolumeProfile) -> tuple[float, float]:
    centres = profile.bin_centres
    w = profile.volume.astype(np.float64, copy=False)
    total = w.sum()
    if total <= 0.0:
        return 0.0, 0.0
    mean = np.sum(w * centres) / total
    var = np.sum(w * (centres - mean) ** 2) / total
    if var <= 0.0:
        return 0.0, 0.0
    std = np.sqrt(var)
    skew = np.sum(w * (centres - mean) ** 3) / total / std**3
    kurt = np.sum(w * (centres - mean) ** 4) / total / std**4 - 3.0
    return float(skew), float(kurt)


def profile_skew(profile: VolumeProfile) -> float:
    """Volume-weighted skewness of price across the profile."""
    return _weighted_moments(profile)[0]


def profile_kurtosis(profile: VolumeProfile) -> float:
    """Volume-weighted excess kurtosis of price across the profile."""
    return _weighted_moments(profile)[1]


def _shape_from(skew: float, profile: VolumeProfile) -> ProfileShape:
    """
    Coarse Market-Profile shape label.

    * ``b`` — selling tail below value (POC in the upper third), left-skewed.
    * ``p`` — buying tail above value (POC in the lower third), right-skewed.
    * ``D`` — balanced, POC near the middle (normal day).
    * ``normal`` — anything that does not meet the above thresholds.
    """
    edges = profile.price_bins
    span = float(edges[-1] - edges[0])
    if span <= 0.0:
        return "normal"
    poc_pos = (profile.poc - edges[0]) / span
    if poc_pos >= 0.66 and skew < -0.2:
        return "b"
    if poc_pos <= 0.34 and skew > 0.2:
        return "p"
    if 0.4 <= poc_pos <= 0.6 and abs(skew) < 0.2:
        return "D"
    return "normal"


def classify_profile_shape(profile: VolumeProfile) -> DistributionStats:
    """Bundle concentration, entropy, skew, kurtosis, and a shape label."""
    skew, kurt = _weighted_moments(profile)
    return DistributionStats(
        gini=volume_concentration(profile),
        entropy=volume_entropy(profile),
        skew=skew,
        kurtosis=kurt,
        shape=_shape_from(skew, profile),
    )
