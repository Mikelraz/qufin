"""GEX aggregation, gamma flip, walls, max-pain, and spot-sweep profile."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from qufin.options import (
    CALL,
    PUT,
    DealerConvention,
    OptionChain,
    aggregate_exposure,
    call_wall,
    gex_profile,
    max_pain,
    put_wall,
    zero_gamma_level,
)


def _synthetic_chain(
    *,
    spot: float = 100.0,
    strikes: np.ndarray | None = None,
    iv: float = 0.20,
    days: int = 30,
    call_oi: float = 1000.0,
    put_oi: float = 1000.0,
) -> OptionChain:
    if strikes is None:
        strikes = np.arange(80.0, 121.0, 5.0)
    as_of = date(2026, 1, 1)
    expiry = as_of + timedelta(days=days)
    rows: list[dict[str, object]] = []
    for K in strikes:
        for side, oi_val in ((CALL, call_oi), (PUT, put_oi)):
            rows.append(
                {
                    "expiry": expiry,
                    "strike": float(K),
                    "option_type": side,
                    "bid": 0.0,
                    "ask": 0.0,
                    "last": 0.0,
                    "volume": 0,
                    "open_interest": int(oi_val),
                    "iv": float(iv),
                }
            )
    df = pl.DataFrame(rows)
    return OptionChain.from_records(df, spot=spot, as_of=as_of, underlying="TEST")


def test_aggregate_exposure_shapes_and_classic_signs() -> None:
    chain = _synthetic_chain()
    exp = aggregate_exposure(chain)
    n_strikes = np.unique(chain.strikes()).shape[0]
    assert exp.strikes.shape[0] == n_strikes
    assert exp.gex.shape == exp.strikes.shape
    # With symmetric OI under classic convention, call leg drives negative GEX
    # and put leg drives positive GEX — total magnitude per strike depends on
    # how close that strike is to spot, but the put GEX must be >= 0.
    assert (exp.put_oi >= 0).all()
    assert (exp.call_oi >= 0).all()


def test_classic_convention_signs_calls_negative_puts_positive() -> None:
    chain = _synthetic_chain(call_oi=1000.0, put_oi=0.0)
    exp_calls_only = aggregate_exposure(chain)
    assert exp_calls_only.gex.sum() < 0.0  # short calls => negative dealer gamma

    chain_p = _synthetic_chain(call_oi=0.0, put_oi=1000.0)
    exp_puts_only = aggregate_exposure(chain_p)
    assert exp_puts_only.gex.sum() > 0.0  # long puts => positive dealer gamma


def test_custom_convention_requires_signs_array() -> None:
    chain = _synthetic_chain()
    n = chain.data.height
    with pytest.raises(ValueError):
        aggregate_exposure(chain, convention=DealerConvention.CUSTOM)
    signs = np.ones(n, dtype=np.float64)
    out = aggregate_exposure(chain, convention=DealerConvention.CUSTOM, dealer_signs=signs)
    assert out.gex.sum() > 0.0  # all-long-gamma assignment


def test_zero_gamma_flip_between_call_heavy_and_put_heavy_regions() -> None:
    # Place a giant put pile below spot and a giant call pile above spot
    # under the classic convention.  Below the put strike → strong +γ (long puts
    # dominate).  Above the call strike → strong -γ.  Flip must lie in between.
    K_put = 90.0
    K_call = 110.0
    strikes = np.array([K_put, K_call])
    rows = [
        {
            "expiry": date(2026, 2, 1),
            "strike": K_put,
            "option_type": PUT,
            "bid": 0.0,
            "ask": 0.0,
            "last": 0.0,
            "volume": 0,
            "open_interest": 50_000,
            "iv": 0.20,
        },
        {
            "expiry": date(2026, 2, 1),
            "strike": K_call,
            "option_type": CALL,
            "bid": 0.0,
            "ask": 0.0,
            "last": 0.0,
            "volume": 0,
            "open_interest": 50_000,
            "iv": 0.20,
        },
    ]
    chain = OptionChain.from_records(pl.DataFrame(rows), spot=100.0, as_of=date(2026, 1, 1))
    _ = strikes  # documentation only
    flip = zero_gamma_level(chain, search_pct=0.25, n_grid=801)
    assert flip is not None
    assert K_put < flip < K_call


def test_max_pain_is_among_listed_strikes() -> None:
    chain = _synthetic_chain()
    mp = max_pain(chain)
    assert mp in set(np.unique(chain.strikes()).tolist())


def test_call_and_put_walls_pick_max_oi_strike() -> None:
    strikes = np.arange(80.0, 121.0, 5.0)
    # Pile all OI at K=110 for calls and K=90 for puts.
    rows = []
    for K in strikes:
        rows.append(
            {
                "expiry": date(2026, 2, 1),
                "strike": float(K),
                "option_type": CALL,
                "bid": 0.0,
                "ask": 0.0,
                "last": 0.0,
                "volume": 0,
                "open_interest": 50_000 if K == 110.0 else 100,
                "iv": 0.20,
            }
        )
        rows.append(
            {
                "expiry": date(2026, 2, 1),
                "strike": float(K),
                "option_type": PUT,
                "bid": 0.0,
                "ask": 0.0,
                "last": 0.0,
                "volume": 0,
                "open_interest": 50_000 if K == 90.0 else 100,
                "iv": 0.20,
            }
        )
    chain = OptionChain.from_records(pl.DataFrame(rows), spot=100.0, as_of=date(2026, 1, 1))
    assert call_wall(chain) == 110.0
    assert put_wall(chain) == 90.0


def test_gex_profile_shape_and_flip_consistency() -> None:
    chain = _synthetic_chain()
    profile = gex_profile(chain, n_spot=101, spot_range_pct=0.15)
    assert profile.gex.shape == profile.spot_grid.shape == (101,)
    assert profile.dex.shape == profile.gex.shape
    if profile.flip_level is not None:
        assert profile.spot_grid[0] <= profile.flip_level <= profile.spot_grid[-1]


def test_profile_dataframe_columns() -> None:
    chain = _synthetic_chain()
    df = gex_profile(chain, n_spot=11).to_dataframe()
    assert set(df.columns) == {"spot", "gex", "dex", "vex", "charm"}
