# qufin.portfolio

Tools for portfolio analysis and mean-variance optimization.

---

## Overview

The `portfolio` subpackage covers the full workflow from raw price data to
optimized weights:

```
prices (polars DataFrame)
  └─ returns.py        → period returns (simple or log)
       └─ covariance.py → (n × n) covariance matrix
       └─ metrics.py    → per-asset or portfolio risk metrics
            └─ optimize.py → optimal weight vectors
```

All optimization inputs are expected in **annualized units**.  Convert
daily covariance with `annualize_cov(daily_cov, 252)` before passing
to any optimizer.

---

## Data Conventions

| Item | Convention |
|---|---|
| Returns | Dimensionless decimals (0.01 = 1 %) |
| Prices DataFrame | One optional `date` column + float columns per asset |
| Returns matrix | NumPy array shape **(T × n)**: rows = time, columns = assets |
| Risk-free rate | Annual rate (e.g. `0.04` for 4 %) |
| `periods_per_year` | 252 daily · 52 weekly · 12 monthly |
| VaR / CVaR | Positive loss magnitudes |

---

## Quick Start

```python
import numpy as np
import polars as pl
import yfinance as yf

from qufin.portfolio import (
    simple_returns, to_returns_matrix, annualized_returns,
    ledoit_wolf_cov, annualize_cov,
    min_variance, max_sharpe, risk_parity, efficient_frontier,
    portfolio_metrics,
)

# 1. Download prices
raw = yf.download(["SPY", "QQQ", "TLT", "GLD"],
                  start="2020-01-01", auto_adjust=True, progress=False)
prices = pl.from_pandas(raw["Close"].reset_index().rename(
    columns={raw["Close"].reset_index().columns[0]: "date"}
))

# 2. Compute returns
ret_df = simple_returns(prices)
mat, names = to_returns_matrix(ret_df)

# 3. Estimate covariance
mu_map = annualized_returns(ret_df, periods_per_year=252)
mu = np.array([mu_map[n] for n in names])
cov = annualize_cov(ledoit_wolf_cov(mat), 252)

# 4. Optimize
ms  = max_sharpe(mu, cov, names, risk_free_rate=0.04)
mv  = min_variance(mu, cov, names)
rp  = risk_parity(cov, names, expected_returns=mu)
ef  = efficient_frontier(mu, cov, names, n_points=60)

# 5. Evaluate in-sample
m = portfolio_metrics(ms.weights, mat, risk_free_rate=0.04)
```

---

## Module Reference

### `returns` — price-to-return transformations

All functions operate on polars DataFrames.  The `date_col` parameter
identifies which column to treat as the time index (default `"date"`);
all other columns are assumed to be asset prices/returns.

| Function | Description |
|---|---|
| `simple_returns(prices, date_col)` | `r_t = P_t / P_{t-1} − 1`. Drops the first row. |
| `log_returns(prices, date_col)` | `r_t = ln(P_t / P_{t-1})`. Additive over time. |
| `cumulative_returns(returns, date_col)` | Running `(1+r_1)·…·(1+r_t) − 1`. |
| `annualize_return(total, n, ppy)` | Compound annualisation: `(1+total)^(ppy/n) − 1`. |
| `annualized_returns(returns, ppy)` | Per-asset geometric annualized returns. |
| `to_returns_matrix(returns, date_col)` | Extract `(T×n)` numpy array + name list. |

---

### `metrics` — risk and performance metrics

All functions accept a 1-D numpy array of **period** returns.

#### Return metrics

| Function | Formula | Notes |
|---|---|---|
| `annualized_volatility(r, ppy)` | `std(r, ddof=1) · sqrt(ppy)` | Sample std |
| `sharpe_ratio(r, rfr, ppy)` | `E[r−rfr/T] / std(r−rfr/T) · sqrt(T)` | Annualized |
| `sortino_ratio(r, rfr, ppy)` | `E[excess] / downside_std · sqrt(T)` | Downside deviation only |
| `calmar_ratio(r, ppy)` | `ann_return / max_drawdown` | Geometric return |

#### Drawdown

| Function | Description |
|---|---|
| `max_drawdown(r)` | Max peak-to-trough fraction of peak wealth. JIT-compiled. |

#### Tail risk

| Function | Formula | Notes |
|---|---|---|
| `historical_var(r, confidence)` | `−quantile_{1−α}(r)` | Non-parametric, positive loss |
| `conditional_var(r, confidence)` | `−E[r \| r ≤ −VaR]` | Expected Shortfall; coherent |

#### Aggregated

```python
m = portfolio_metrics(weights, returns_matrix, risk_free_rate=0.04)
# keys: annualized_return, annualized_volatility, sharpe_ratio,
#        sortino_ratio, max_drawdown, calmar_ratio, var_95, cvar_95
```

---

### `covariance` — covariance estimation

All estimators accept a **(T × n)** returns matrix and return a
**per-period (n × n)** covariance matrix.  Pass through `annualize_cov`
before optimization.

#### Estimators

**`sample_cov(returns, ddof=1)`**
Classical unbiased estimator.  Noisy when `n/T > 0.1` (many assets,
short history).

**`ledoit_wolf_cov(returns)`**
Analytical shrinkage toward a scaled identity target `μI`:

```
Σ̂ = (1 − α) · S  +  α · μ · I
μ = tr(S) / n
α = min(1, β² / δ²)
```

- `δ²`: Frobenius distance of `S` from the target (how much shrinkage is possible)
- `β²`: sampling error in `S` (how much shrinkage is needed)
- A larger `α` means more regularization toward equal, uncorrelated variances.

Best choice when `T` is small relative to `n` or when assets are highly correlated.

Reference: Ledoit & Wolf (2004), *Journal of Multivariate Analysis*, 88(2).

**`ewm_cov(returns, halflife)`**
Exponentially weighted, useful in regime-changing markets.  The halflife
controls how fast old observations fade:

```
w_i ∝ (1−α)^i,   α = 1 − exp(−ln2 / halflife)
```

Typical halflives: 21 (1-month), 63 (quarter), 126 (6-month).

#### Utilities

| Function | Description |
|---|---|
| `annualize_cov(cov, ppy)` | Multiply by `periods_per_year` (i.i.d. variance scaling). |
| `cov_to_corr(cov)` | Normalize to correlation matrix (ones on diagonal). |

---

### `optimize` — portfolio optimizers

All optimizers use SLSQP with a full-investment constraint (`sum(w) = 1`).
Inputs must be in **annualized units**.

#### `min_variance`

```
min  w'Σw    s.t.  Σw_i = 1,  w_i ≥ 0
```

The safest optimizer: does not depend on expected return estimates.
Sits at the left tip of the efficient frontier.

#### `max_sharpe`

```
max  (w'μ − r_f) / sqrt(w'Σw)    s.t.  Σw_i = 1,  w_i ≥ 0
```

Non-convex; highly sensitive to the expected return vector.  Tends to
concentrate heavily in a few assets.  Use when you have conviction in `μ`.

#### `efficient_return`

```
min  w'Σw    s.t.  Σw_i = 1,  w'μ = target,  w_i ≥ 0
```

One point on the efficient frontier at a given return level.

#### `efficient_frontier`

Sweeps `n_points` return targets between the min-variance return and
`max(μ)`, calling `efficient_return` at each step.  Returns an
`EfficientFrontier` container with arrays of `returns`, `volatilities`,
`sharpe_ratios`, and `weights`.

#### `risk_parity`

Equalizes each asset's fractional contribution to total portfolio variance:

```
RC_i = w_i · (Σw)_i / (w'Σw) = 1/n  for all i
```

Does not use expected returns; well-suited when return forecasts are
unreliable.  Robust alternative to `max_sharpe`.

---

## Result Objects

### `OptimizationResult`

| Field | Type | Description |
|---|---|---|
| `weights` | `NDArray[float64]` shape (n,) | Portfolio weights summing to 1 |
| `asset_names` | `list[str]` | Asset identifiers |
| `expected_return` | `float` | `w'μ` (annualized) |
| `expected_volatility` | `float` | `sqrt(w'Σw)` (annualized) |
| `sharpe_ratio` | `float` | `(E[r] − rfr) / σ` |
| `success` | `bool` | SLSQP convergence flag |
| `message` | `str` | Solver status string |

```python
result.as_dict()  # → {"SPY": 0.42, "QQQ": 0.0, ...}
```

### `EfficientFrontier`

| Field | Type | Description |
|---|---|---|
| `returns` | `NDArray` shape (k,) | Annualized expected returns |
| `volatilities` | `NDArray` shape (k,) | Annualized volatilities |
| `sharpe_ratios` | `NDArray` shape (k,) | Sharpe ratios |
| `weights` | `NDArray` shape (k, n) | Weight matrix per frontier point |
| `asset_names` | `list[str]` | Asset identifiers |
| `risk_free_rate` | `float` | Rate used for Sharpe computation |

---

## CLI Script

`scripts/portfolio_optimize.py` runs the full pipeline from the command line:

```
python scripts/portfolio_optimize.py <tickers> [options]

Options:
  --start DATE        Start date (default: 2020-01-01)
  --end DATE          End date (default: today)
  --method            min-variance | max-sharpe | risk-parity | all
  --cov               sample | ledoit-wolf | ewm
  --rfr FLOAT         Annual risk-free rate (default: 0.04)
  --ppy INT           Periods per year (default: 252)
  --ewm-halflife FLOAT  EWM halflife in days (default: 63)
  --plot              Show efficient frontier plot
```

Example:

```
python scripts/portfolio_optimize.py SPY QQQ TLT GLD \
    --start 2020-01-01 --rfr 0.04 --cov ledoit-wolf --plot
```
