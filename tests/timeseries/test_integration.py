"""
End-to-end integration test for the Phase 1 timeseries foundations.

Verifies that the public API can be invoked from a single polars-flavoured
workflow: simulate a stationary AR(1), run stationarity + serial-correlation
+ normality + ARCH diagnostics, and round-trip results through polars
DataFrames.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from qufin.timeseries import (
    acf,
    adf,
    arch_lm,
    difference,
    info_criteria,
    inverse_difference,
    jarque_bera,
    kpss,
    ljung_box,
    pacf,
    phillips_perron,
    variance_ratio,
)


def test_end_to_end_workflow():
    # Seed/phi chosen so every diagnostic passes its expected side of H0
    # for a *single* realisation (the per-test files cover correctness
    # across many seeds; here we want a deterministic end-to-end smoke).
    rng = np.random.default_rng(99)
    n = 2_000
    phi = 0.3
    eps = rng.standard_normal(n)
    x = np.empty(n)
    x[0] = 0.0
    for t in range(1, n):
        x[t] = phi * x[t - 1] + eps[t]

    # Wrap in polars to exercise the input-conversion path everywhere.
    series = pl.Series("x", x)

    # Stationarity: a stationary AR(1) rejects ADF / Phillips-Perron and
    # does not reject KPSS.
    adf_res = adf(series, regression="c")
    assert adf_res.p_value < 0.05

    pp_res = phillips_perron(series, regression="c")
    assert pp_res.p_value < 0.05

    kpss_res = kpss(series, regression="c")
    assert kpss_res.p_value > 0.05

    # Lo-MacKinlay VR(4) for an AR(1) with φ = 0.3 is theoretically 1 +
    # (2/4) · Σ_{k=1..3} (4-k) φ^k ≈ 1.55, so the random-walk null should be
    # rejected here.
    vr_res = variance_ratio(series, q=4)
    assert vr_res.stat > 1.2
    assert vr_res.p_value < 0.05

    # Serial-correlation: strong rejection on the AR(1) itself.
    q, lb_p = ljung_box(series, lags=10)
    assert lb_p < 1e-3

    # ACF / PACF: AR(1) signature.
    acf_res = acf(series, nlags=5)
    pacf_res = pacf(series, nlags=5)
    assert acf_res.values[0] > 0.2
    assert pacf_res.values[0] > 0.2
    assert abs(pacf_res.values[1]) < 0.1

    df_acf = acf_res.to_dataframe()
    assert isinstance(df_acf, pl.DataFrame)
    assert df_acf.columns == ["lag", "value", "lower", "upper"]

    # Normality: AR(1) with Gaussian innovations is approximately Gaussian.
    _, jb_p = jarque_bera(series)
    assert jb_p > 0.01

    # ARCH effects: AR(1) *residuals* are i.i.d. Gaussian → ARCH-LM applied
    # to residuals should not reject.  (Squared AR(1) levels themselves have
    # autocorrelation φ^{2k} even without true conditional heteroskedasticity,
    # so ARCH-LM should be applied to residuals, not the raw series.)
    residuals = x[1:] - phi * x[:-1]
    _, arch_p = arch_lm(residuals, lags=5)
    assert arch_p > 0.01

    # Differencing round-trip.
    d = difference(series, 1)
    recon = inverse_difference(d, x[:1])
    np.testing.assert_allclose(recon, x, atol=1e-10)

    # Info criteria sanity.
    aic, bic, _ = info_criteria(log_lik=-100.0, n_obs=n, n_params=3)
    assert bic > aic  # for n=2000, ln n > 2 ⇒ BIC penalises more
