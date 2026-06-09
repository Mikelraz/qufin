"""Tests for qufin.fundamentals.scores."""

from __future__ import annotations

import math

import pytest

from qufin.fundamentals import (
    FundamentalSnapshot,
    altman_z_double_prime,
    altman_z_score,
    beneish_m_score,
    piotroski_f_score,
)


def test_piotroski_perfect_score(
    improving_pair: tuple[FundamentalSnapshot, FundamentalSnapshot],
) -> None:
    curr, prev = improving_pair
    f = piotroski_f_score(curr, prev)
    assert f.score == 9
    assert all(f.as_dict().values())


def test_piotroski_deteriorating_firm(
    improving_pair: tuple[FundamentalSnapshot, FundamentalSnapshot],
) -> None:
    curr, prev = improving_pair
    # Swap roles: the weaker year is now "current"
    f = piotroski_f_score(prev, curr)
    assert f.score == 4
    assert f.positive_net_income
    assert not f.rising_roa
    assert not f.falling_leverage


def test_altman_z_safe(base_snapshot: FundamentalSnapshot) -> None:
    z = altman_z_score(base_snapshot)
    assert z.variant == "z"
    assert z.score == pytest.approx(3.99)
    assert z.zone == "safe"


def test_altman_z_distress() -> None:
    distressed = FundamentalSnapshot(
        current_assets=200.0,
        current_liabilities=500.0,
        total_assets=1000.0,
        retained_earnings=-200.0,
        ebit=10.0,
        market_cap=100.0,
        total_liabilities=900.0,
        revenue=300.0,
    )
    z = altman_z_score(distressed)
    assert z.zone == "distress"
    assert z.score < 1.81


def test_altman_z_double_prime_safe(base_snapshot: FundamentalSnapshot) -> None:
    z = altman_z_double_prime(base_snapshot)
    assert z.variant == "z_double_prime"
    # 6.56*0.2 + 3.26*0.3 + 6.72*0.1 + 1.05*(800/1200)
    assert z.score == pytest.approx(3.662, abs=1e-3)
    assert z.zone == "safe"


def test_altman_nan_is_distress() -> None:
    z = altman_z_score(FundamentalSnapshot())
    assert math.isnan(z.score)
    assert z.zone == "distress"


def _beneish_prev() -> FundamentalSnapshot:
    return FundamentalSnapshot(
        revenue=1000.0,
        cogs=600.0,
        receivables=100.0,
        current_assets=400.0,
        ppe_net=500.0,
        total_assets=1000.0,
        depreciation=50.0,
        sga_expense=100.0,
        current_liabilities=200.0,
        long_term_debt=300.0,
        net_income=80.0,
        operating_cash_flow=80.0,
    )


def test_beneish_clean_firm_not_flagged() -> None:
    prev = _beneish_prev()
    curr = FundamentalSnapshot(
        revenue=1100.0,
        cogs=660.0,
        receivables=110.0,
        current_assets=440.0,
        ppe_net=550.0,
        total_assets=1100.0,
        depreciation=55.0,
        sga_expense=110.0,
        current_liabilities=220.0,
        long_term_debt=330.0,
        net_income=88.0,
        operating_cash_flow=88.0,
    )
    m = beneish_m_score(curr, prev)
    assert m.sgi == pytest.approx(1.1)
    assert m.dsri == pytest.approx(1.0)
    assert m.gmi == pytest.approx(1.0)
    assert m.tata == pytest.approx(0.0)
    assert not m.manipulator
    assert m.score < -1.78


def test_beneish_manipulator_flagged() -> None:
    prev = _beneish_prev()
    curr = FundamentalSnapshot(
        revenue=1500.0,
        cogs=1100.0,
        receivables=300.0,
        current_assets=500.0,
        ppe_net=700.0,
        total_assets=1600.0,
        depreciation=40.0,
        sga_expense=120.0,
        current_liabilities=400.0,
        long_term_debt=600.0,
        net_income=200.0,
        operating_cash_flow=20.0,
    )
    m = beneish_m_score(curr, prev)
    assert m.dsri == pytest.approx(2.0)
    assert m.sgi == pytest.approx(1.5)
    assert m.manipulator
    assert m.score > -1.78


def test_beneish_score_consistent_with_indices() -> None:
    prev = _beneish_prev()
    curr = FundamentalSnapshot(
        revenue=1500.0,
        cogs=1100.0,
        receivables=300.0,
        current_assets=500.0,
        ppe_net=700.0,
        total_assets=1600.0,
        depreciation=40.0,
        sga_expense=120.0,
        current_liabilities=400.0,
        long_term_debt=600.0,
        net_income=200.0,
        operating_cash_flow=20.0,
    )
    m = beneish_m_score(curr, prev)
    expected = (
        -4.84
        + 0.92 * m.dsri
        + 0.528 * m.gmi
        + 0.404 * m.aqi
        + 0.892 * m.sgi
        + 0.115 * m.depi
        - 0.172 * m.sgai
        + 4.679 * m.tata
        - 0.327 * m.lvgi
    )
    assert m.score == pytest.approx(expected)


def test_beneish_nan_not_manipulator() -> None:
    m = beneish_m_score(FundamentalSnapshot(), FundamentalSnapshot())
    assert math.isnan(m.score)
    assert not m.manipulator
