# qufin tutorials — a learning path

A numbered, **basic → advanced** curriculum for quantitative finance with the
`qufin` toolkit. Every notebook is **self-contained and executed** — open it and
read top to bottom. Work straight through for a full course, or jump to a number
and follow the "what next" pointers.

Examples use *synthetic data that provably fits the model* (simulate → recover
the known parameters) and/or *specifically chosen real data* (pulled via
`yfinance`, with an offline synthetic fallback so every notebook runs without a
network).

## How to run

```bash
uv sync --all-groups
uv run jupyter lab            # then open notebooks/ in number order
```

Each numbered notebook `NN_<name>.ipynb` is built from a generator script
`_gen_NN_<name>.py` (the pure-math notebooks 02–05 and the options notebooks
25–27 are hand-authored and have no generator). To regenerate and re-execute one
after a source change:

```bash
uv run python notebooks/_gen_15_garch_volatility.py
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/15_garch_volatility.ipynb
```

---

## Part I · Foundations — math, statistics, finance

The toolkit every later notebook assumes. Start here if you're new.

| # | Notebook | What you'll learn |
|---|---|---|
| 01 | [Financial markets & returns](01_financial_markets_and_returns.ipynb) | instruments, simple vs **log returns**, compounding, the √T rule, fat tails & volatility clustering, drawdown & Sharpe |
| 02 | [Probability & statistics](02_probability_and_statistics.ipynb) | distributions, estimation, hypothesis testing |
| 03 | [Linear algebra](03_linear_algebra.ipynb) | vectors, matrices, eigen-decompositions for finance |
| 04 | [Calculus](04_calculus.ipynb) | derivatives, gradients, Taylor expansion |
| 05 | [Differential equations](05_differential_equations.ipynb) | ODEs for continuous-time finance |
| 06 | [Regression & econometrics](06_regression_and_econometrics.ipynb) | OLS & inference, **CAPM beta**, multicollinearity, the **spurious-regression** trap |
| 07 | [Optimization](07_optimization.ipynb) | gradient descent, root finding, Lagrange/SLSQP, mean-variance, **MLE as optimization** |
| 08 | [Stochastic processes](08_stochastic_processes.ipynb) | random walks, **Brownian motion**, GBM, Itô & the volatility drag, OU mean reversion |
| 09 | [Monte Carlo & numerical methods](09_monte_carlo_methods.ipynb) | MC integration, option pricing by simulation, variance reduction, **implied vol** by root-finding, finite-difference Greeks |

## Part II · Market data & technical analysis

Reading the chart: indicators, where volume trades, and trade-level structure.

| # | Notebook | `qufin` module |
|---|---|---|
| 10 | [Technical indicators](10_technical_indicators.ipynb) | `qufin.indicators` — moving averages, momentum, trend, volatility, volume |
| 11 | [Volume distribution & market profile](11_volume_distribution.ipynb) | `qufin.volume_distribution` — volume profile, VWAP bands, TPO, CVD/delta |
| 12 | [Market microstructure](12_market_microstructure.ipynb) | `qufin.microstructure` — trade classification, spread estimators, price impact, VPIN |
| 13 | [Wyckoff method](13_wyckoff_method.ipynb) | `qufin.wyckoff` — accumulation/distribution phases, effort vs result |

## Part III · Time series & volatility

Modelling the conditional mean and — especially — the conditional variance.

| # | Notebook | `qufin` module |
|---|---|---|
| 14 | [Stationarity & ARIMA forecasting](14_arima_forecasting.ipynb) | `qufin.timeseries` — unit-root tests, ACF/PACF, AR/MA/ARMA/ARIMA/SARIMA, Diebold–Mariano |
| 15 | [GARCH volatility](15_garch_volatility.ipynb) | `qufin.timeseries.garch` — GARCH/EGARCH/GJR/EWMA, leverage effect, DCC correlation |
| 16 | [Realized volatility](16_realized_volatility.ipynb) | `qufin.timeseries.realized` — realized variance, bipower variation, HAR-RV |
| 17 | [Long memory & fractals](17_long_memory.ipynb) | `qufin.timeseries.fractal` — Hurst exponent, fractal dimension |

## Part IV · Trading signals & regimes

Turning models into positions; detecting the state of the market.

| # | Notebook | `qufin` module |
|---|---|---|
| 18 | [Momentum factors](18_momentum.ipynb) | `qufin.analysis.momentum` — time-series & cross-sectional momentum, vol targeting |
| 19 | [Mean reversion & Ornstein–Uhlenbeck](19_mean_reversion_ou.ipynb) | `qufin.models` — OU process, half-life, z-score, `MeanReversionStrategy` |
| 20 | [Cointegration & pairs trading](20_cointegration_pairs.ipynb) | `qufin.timeseries.cointegration` — Engle–Granger/Johansen, Kalman hedge, pairs strategy |
| 21 | [Markov chains & HMM](21_markov_hmm.ipynb) | `qufin.markov` — Markov/higher-order chains, discrete HMM (Baum–Welch/Viterbi) |
| 22 | [Regime-switching models](22_regime_switching.ipynb) | `qufin.timeseries.regime` — Markov-switching AR, Hamilton filter, regime strategy |

## Part V · Cross-section & portfolio

Picking and combining assets.

| # | Notebook | `qufin` module |
|---|---|---|
| 23 | [Fundamental analysis](23_fundamental_analysis.ipynb) | `qufin.fundamentals` — ratios, multiples, valuation, quality scores, screening |
| 24 | [Portfolio optimization](24_portfolio_optimization.ipynb) | `qufin.portfolio` — mean-variance, efficient frontier, risk parity, shrinkage, out-of-sample fragility |

## Part VI · Options & derivatives

Pricing, the Greeks, and the dealer-positioning lens.

| # | Notebook | `qufin` module |
|---|---|---|
| 25 | [Options Greeks](25_options_greeks.ipynb) | `qufin.options.greeks` — delta, gamma, vega, theta, rho |
| 26 | [Options pricing & implied vol](26_options_pricing.ipynb) | `qufin.options` — pricing, implied volatility, the full options stack |
| 27 | [Gamma exposure (GEX)](27_gamma_exposure.ipynb) | `qufin.options.gex` — dealer gamma exposure, flip point, walls |

---

**A note on the examples.** Strategy Sharpe ratios in the application notebooks are
mostly **in-sample** (or pre-cost) and exist to illustrate the mechanics, not to
advertise performance. For honest evaluation use the walk-forward tooling in
`qufin.trading.training` and net returns against realistic transaction costs.
