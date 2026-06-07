# qufin tutorials

A guided tour of the `qufin` toolkit. Each notebook is **self-contained and
executed** — open it and read top to bottom. Examples use *synthetic data that
provably fits the model* (simulate → recover the known parameters) and/or
*specifically chosen real data* (pulled via `yfinance`, with an offline
synthetic fallback so every notebook runs without a network).

## How to run

```bash
uv sync --all-groups
uv run jupyter lab          # then open any notebook under notebooks/
```

Most package tutorials are generated from a script so they stay in sync with the
library. To regenerate one after a source change:

```bash
uv run python notebooks/_gen_<area>_tutorial.py
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/<area>_tutorial.ipynb
```

## Suggested learning path

### 0 · Mathematical foundations
Start here if you want the machinery behind everything else.

| Notebook | Topic |
|---|---|
| [01_probability_statistics_basics](01_probability_statistics_basics.ipynb) | distributions, estimation, hypothesis testing |
| [02_linear_algebra_for_finance](02_linear_algebra_for_finance.ipynb) | vectors, matrices, eigen-decompositions |
| [03_calculus_refresher](03_calculus_refresher.ipynb) | derivatives, optimisation, Taylor expansion |
| [04_ordinary_differential_equations](04_ordinary_differential_equations.ipynb) | ODEs for continuous-time finance |

### 1 · Indicators & price structure
Reading the chart: classic technicals, support/resistance, and where volume trades.

| Notebook | `qufin` module | Covers |
|---|---|---|
| [indicators_package_tutorial](indicators_package_tutorial.ipynb) | `qufin.indicators` | moving averages, momentum, trend, volatility, volume |
| [wyckoff_package_tutorial](wyckoff_package_tutorial.ipynb) | `qufin.wyckoff` | accumulation/distribution phases, effort-vs-result |
| [volume_distribution_tutorial](volume_distribution_tutorial.ipynb) | `qufin.volume_distribution` | volume profile, VWAP bands, Market Profile/TPO, CVD/delta |
| [momentum_package_tutorial](momentum_package_tutorial.ipynb) | `qufin.analysis.momentum` | time-series & cross-sectional momentum, vol targeting |

### 2 · Volatility & time-series forecasting
Modelling the conditional mean and (especially) the conditional variance.

| Notebook | `qufin` module | Covers |
|---|---|---|
| [arima_forecasting_tutorial](arima_forecasting_tutorial.ipynb) | `qufin.timeseries` | stationarity (ADF/KPSS/PP/VR), ACF/PACF, AR/MA/ARMA/ARIMA/SARIMA, forecast eval |
| [garch_volatility_tutorial](garch_volatility_tutorial.ipynb) | `qufin.timeseries.garch` | GARCH/EGARCH/GJR/EWMA, leverage effect, DCC correlation |
| [realized_volatility_package_tutorial](realized_volatility_package_tutorial.ipynb) | `qufin.timeseries.realized` | realized variance, bipower variation, HAR-RV |
| [long_memory_package_tutorial](long_memory_package_tutorial.ipynb) | `qufin.timeseries.fractal` | Hurst exponent, fractal dimension, long memory |

### 3 · Mean reversion & statistical arbitrage
Trading the gap between a series and its fair value.

| Notebook | `qufin` module | Covers |
|---|---|---|
| [ou_mean_reversion_tutorial](ou_mean_reversion_tutorial.ipynb) | `qufin.models` | Ornstein-Uhlenbeck, half-life, z-score, `MeanReversionStrategy` |
| [cointegration_pairs_tutorial](cointegration_pairs_tutorial.ipynb) | `qufin.timeseries.cointegration` | Engle-Granger/Johansen, Kalman hedge ratio, `CointegrationPairsStrategy` |

### 4 · Regimes & state models
Markets switch states; here is how to find the switch.

| Notebook | `qufin` module | Covers |
|---|---|---|
| [markov_hmm_tutorial](markov_hmm_tutorial.ipynb) | `qufin.markov` | Markov chains, higher-order chains, discrete HMM (Baum-Welch/Viterbi) |
| [regime_switching_tutorial](regime_switching_tutorial.ipynb) | `qufin.timeseries.regime` | Markov-switching AR, Hamilton filter, `RegimeSwitchingStrategy` |

### 5 · Microstructure & order flow
What the tape says at the trade level.

| Notebook | `qufin` module | Covers |
|---|---|---|
| [microstructure_package_tutorial](microstructure_package_tutorial.ipynb) | `qufin.microstructure` | trade classification, spread estimators, price impact, VPIN |

### 6 · Portfolio construction
Combining assets into a book.

| Notebook | `qufin` module | Covers |
|---|---|---|
| [portfolio_optimization_tutorial](portfolio_optimization_tutorial.ipynb) | `qufin.portfolio` | mean-variance, efficient frontier, min-var/max-Sharpe/risk-parity, shrinkage, out-of-sample fragility |

### 7 · Options & derivatives
Pricing and the dealer-positioning lens.

| Notebook | `qufin` module | Covers |
|---|---|---|
| [options_greeks](options_greeks.ipynb) | `qufin.options.greeks` | delta, gamma, vega, theta, rho |
| [options_package_tutorial](options_package_tutorial.ipynb) | `qufin.options` | pricing, implied volatility, the full options stack |
| [gamma_exposure_gex](gamma_exposure_gex.ipynb) | `qufin.options.gex` | dealer gamma exposure, flip point, walls |

---

**A note on the examples.** Backtest Sharpe ratios shown in the strategy
sections are mostly **in-sample** (or pre-cost) and are there to illustrate the
mechanics, not to advertise performance. For honest evaluation use the
walk-forward tooling in `qufin.trading.training` and net returns against
realistic transaction costs.
