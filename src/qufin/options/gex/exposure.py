"""
Dealer-position exposure aggregation per strike.

All exposures are reported in *dollars per 1% spot move* (the SqueezeMetrics
convention).  Convert to "shares of underlying" by dividing by ``S²·0.01``.
"""

from __future__ import annotations

from enum import StrEnum

import numpy as np

from .._types import CALL, OptionChain, StrikeExposure
from ..greeks import greeks_for_chain


class DealerConvention(StrEnum):
    """How to assign a dealer-position sign to each contract."""

    CLASSIC = "classic"  # dealers short calls (-1), long puts (+1)
    CUSTOM = "custom"  # caller supplies per-contract signs


def dealer_signs(
    is_call: np.ndarray, *, convention: DealerConvention = DealerConvention.CLASSIC
) -> np.ndarray:
    """Return ±1 per contract under the chosen dealer-positioning convention."""
    match convention:
        case DealerConvention.CLASSIC:
            return np.where(is_call, -1.0, 1.0)
        case DealerConvention.CUSTOM:
            raise ValueError(
                "DealerConvention.CUSTOM requires an explicit signs array — "
                "pass it via aggregate_exposure(..., dealer_signs=...)."
            )


def _per_contract_exposures(
    chain: OptionChain,
    *,
    convention: DealerConvention,
    signs: np.ndarray | None,
    sigma: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    g = greeks_for_chain(chain, sigma=sigma)
    is_call = chain.is_call()
    oi = chain.open_interest()
    if convention is DealerConvention.CUSTOM:
        if signs is None or signs.shape[0] != is_call.shape[0]:
            raise ValueError("custom convention requires a signs array of the chain length")
        sgn = signs.astype(np.float64)
    else:
        sgn = dealer_signs(is_call, convention=convention)

    notional = chain.multiplier * oi * sgn
    s_sq_pct = chain.spot * chain.spot * 0.01

    gex = g.gamma * notional * s_sq_pct
    dex = g.delta * notional * chain.spot
    vex = g.vanna * notional * chain.spot
    charm = g.charm * notional * chain.spot
    return gex, dex, vex, charm, is_call.astype(np.float64)


def aggregate_exposure(
    chain: OptionChain,
    *,
    convention: DealerConvention = DealerConvention.CLASSIC,
    dealer_signs: np.ndarray | None = None,
    sigma: np.ndarray | None = None,
) -> StrikeExposure:
    """
    Aggregate dealer GEX/DEX/VEX/charm per strike across all expiries.

    Parameters
    ----------
    chain         OptionChain snapshot.
    convention    Dealer-sign convention (classic or custom).
    dealer_signs  Per-contract ±1 array; required when ``convention=CUSTOM``.
    sigma         Override implied vols (defaults to ``chain.iv``).
    """
    gex_c, dex_c, vex_c, charm_c, is_call = _per_contract_exposures(
        chain, convention=convention, signs=dealer_signs, sigma=sigma
    )
    strikes = chain.strikes()
    oi = chain.open_interest()

    uniq_strikes, inverse = np.unique(strikes, return_inverse=True)
    n = uniq_strikes.shape[0]

    gex_k = np.zeros(n, dtype=np.float64)
    dex_k = np.zeros(n, dtype=np.float64)
    vex_k = np.zeros(n, dtype=np.float64)
    charm_k = np.zeros(n, dtype=np.float64)
    call_oi = np.zeros(n, dtype=np.float64)
    put_oi = np.zeros(n, dtype=np.float64)

    np.add.at(gex_k, inverse, gex_c)
    np.add.at(dex_k, inverse, dex_c)
    np.add.at(vex_k, inverse, vex_c)
    np.add.at(charm_k, inverse, charm_c)
    np.add.at(call_oi, inverse, oi * is_call)
    np.add.at(put_oi, inverse, oi * (1.0 - is_call))

    return StrikeExposure(
        strikes=uniq_strikes,
        gex=gex_k,
        dex=dex_k,
        vex=vex_k,
        charm=charm_k,
        call_oi=call_oi,
        put_oi=put_oi,
        notes={
            "total_gex": float(gex_k.sum()),
            "total_dex": float(dex_k.sum()),
            "spot": float(chain.spot),
        },
    )


def signed_call_put_split(
    chain: OptionChain, *, sigma: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return ``(strikes, call_gex_signed, put_gex_signed)`` under the classic
    convention.  Useful for plotting call (red) vs. put (green) GEX bars.
    """
    g = greeks_for_chain(chain, sigma=sigma)
    is_call = chain.is_call()
    oi = chain.open_interest()
    sgn = np.where(is_call, -1.0, 1.0)
    contract_gex = g.gamma * oi * chain.multiplier * sgn * chain.spot * chain.spot * 0.01

    strikes = chain.strikes()
    uniq, inverse = np.unique(strikes, return_inverse=True)
    n = uniq.shape[0]
    call_gex = np.zeros(n, dtype=np.float64)
    put_gex = np.zeros(n, dtype=np.float64)
    mask_call = chain.data["option_type"].to_numpy() == CALL
    np.add.at(call_gex, inverse[mask_call], contract_gex[mask_call])
    np.add.at(put_gex, inverse[~mask_call], contract_gex[~mask_call])
    return uniq, call_gex, put_gex
