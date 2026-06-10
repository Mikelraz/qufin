# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

High-performance Python toolkit for quantitative analysis and modelling of financial market products. Used to develop, train, and test trading strategies and models. Library code lives in `src/qufin/`; research notes are Jupyter notebooks in `notebooks/`; CLI tools and demos in `scripts/`; tests mirror the package tree under `tests/`.

---

## 1. Tooling (Non-Negotiable)

| Tool | Purpose | Never use instead |
|---|---|---|
| `uv` | Package management | `pip`, `poetry`, `conda` |
| `ruff` | Linting and formatting | `flake8`, `black`, `isort` |
| `pytest` + `pytest-xdist` | Testing | `unittest` |
| `pyright` | Type checking | `mypy` |

All dependencies are declared in `pyproject.toml`. Dev dependencies are in the `[dependency-groups] dev` group; live-broker dependencies (`alpaca-py`, `ib-async`, and the sibling `trade-republic-api` via a `[tool.uv.sources]` path) are in the optional `trading-live` group and are **not** installed by default.

---

## 2. Standard Commands

```bash
uv sync --all-groups                    # install all deps including dev
uv sync --group trading-live            # add live-broker deps (alpaca-py, ib-async, trade-republic-api)
uv add <package>                        # add a runtime dependency
uv add --group dev <package>            # add a dev dependency

ruff check --fix .                      # lint and auto-fix
ruff format .                           # format

pytest -n auto                          # run tests across all cores (the default addopts)
pytest tests/wyckoff/test_swings.py     # run one test file
pytest tests/wyckoff/test_swings.py::test_name   # run one test
pytest -k "swing and not zigzag"        # run by keyword
pytest -n auto --cov=src                # with coverage

pyright                                 # strict type-check (scoped to src/ only)

python -m cProfile -s time <script>     # profile a script
```

`testpaths` is `tests/` and `-n auto` is baked into `addopts`, so a bare `pytest` already parallelizes.

---

## 3. Architecture

`src/qufin/` is a flat collection of **domain subpackages**, each self-contained and independently importable. There is no central orchestrator — `qufin/__init__.py` is intentionally empty of re-exports; import from the subpackage (`from qufin.wyckoff import ...`).

### Subpackage map

| Package | Responsibility |
|---|---|
| `data/` | Market-data layer: schemas, `vendors/` (yfinance, alpaca, ibkr, csv), `calendars/` (NYSE), `store/` (partitioned Parquet + manifest), `adjustments/` (splits/dividends), alternative `bars/` (volume/imbalance/run bars), `universe/` (PIT index membership, survivorship). Entry point: `DataPipeline`. |
| `indicators/` | Classic TA: moving averages, momentum, trend, volatility, volume, support/resistance. |
| `volume_distribution/` | VWAP, volume/TPO profiles, delta, order-flow stats. |
| `microstructure/` | Trade classification, spread estimators, price impact, order flow, VPIN/PIN. |
| `wyckoff/` | Wyckoff-method analysis: swings, trading ranges, climax/spring events, phase classifier (rule-based + HMM), point-and-figure. |
| `timeseries/` | ARIMA, GARCH family (`garch/`: garch/egarch/gjr/ewma/dcc), `cointegration/` (Engle-Granger, Johansen, VECM), Kalman/state-space, stationarity, realized vol, fractal/long-memory, regime detection. |
| `markov/` | Markov chains, higher-order chains, HMM, Gaussian HMM. |
| `models/` | Stochastic models: OU process, spread models. |
| `options/` | Black-Scholes `pricing`, `greeks`, implied vol (`iv`), and `gex/` (gamma-exposure profile, flip point, walls). |
| `portfolio/` | Returns, covariance estimation (sample/Ledoit-Wolf/EWM), risk metrics, mean-variance/risk-parity optimizers. See `docs/portfolio.md`. |
| `fundamentals/` | Financial-statement ratios, multiples, growth, valuation, scores, screening. |
| `analysis/` | Cross-sectional studies (momentum, cointegration screen) built on the above. |
| `strategies/` | Concrete signal-generating strategies (hull suite, confluence, EMA-cross, mean-reversion, regime-switching, TS-momentum). |
| `trading/` | Backtest + live execution framework (see below). |

### Recurring per-package conventions (learn these once, apply everywhere)

- **`_types.py`** — dataclasses, `TypeAlias`es, and polars schema constants for the package. Schemas like `BAR_SCHEMA` / `OHLCV` define the canonical column contract.
- **`_kernels.py`** — `@njit(cache=True)` numba hot loops. **Only inherently-recursive or hard-to-vectorise routines live here**; anything that maps cleanly to a windowed numpy/polars expression stays in the public module. Keep kernels free of Python objects so they stay nopython-compilable.
- **`_util.py` / `_io.py` / `_likelihood.py`** — private helpers; the leading underscore marks "not part of the public API".
- **`data/` subdirectory inside a package** (e.g. `options/data/`, `fundamentals/data/`) — vendor-specific fetching, kept separate from the analytics that consume it.
- **`__init__.py`** is the public surface: it has a module-map docstring + a quick-start example, and re-exports the package's API via `__all__`. When adding a public symbol, wire it into both the imports and `__all__`.
- Data fetching, feature engineering, and model logic stay in **distinct modules** — do not mix a yfinance call into a pricing function.

### The `trading/` framework

This is the most layered subpackage. The design separates *signal logic* (portable across backtest and live) from *execution*.

- `strategy/base.py` — the `Strategy` `Protocol` (`on_start`/`on_bar`/`on_fill`/`on_end`) and `StrategyContext` (read-only per-bar view of account, positions, history). Subclass `StrategyBase` for no-op defaults. `strategy/adapters.py` lifts `strategies/` signal generators onto this protocol.
- `engine/` — event-driven backtest. `BacktestEngine` drives a `Strategy` over a `Clock` of bars, routes orders through an `ExecutionModel` (default: next-bar-open fills), and accumulates a `Portfolio` into a `BacktestReport`. `options_engine.py` handles option contracts.
  - **Look-ahead invariant (do not break this):** at step `t` the strategy sees `history[:t+1]`; orders emitted at `t` only fill at `t+1`. `tests/trading/engine/test_lookahead.py` guards it.
- `brokers/` — the `Broker` `Protocol` (async `connect`/`account`/`place_order`/`stream_bars`/…) with `paper`, `alpaca`, `ibkr`, and `trade_republic` implementations. Strategy code is identical across brokers because every call returns the canonical `Order`/`Fill`/`Position`/`AccountSnapshot` from `trading/_types.py`. Live trading is asyncio; the backtest engine is synchronous. `trade_republic.py` (`TradeRepublicBroker`) wraps the sibling `trade-republic-api` client; account/positions/limit+stop orders are solid, while `stream_bars`/`stream_fills` are best-effort polling (Trade Republic exposes ticks, not bars, and has no native fills push).
- `training/` — parameter search (`search.py`, `objectives.py`), `walk_forward.py` splits, and an `ml/` pipeline (feature extraction → signal model).
- `evaluation/` — `tearsheet`, `attribution`, `compare`.

---

## 4. Performance Mandates

### Data
- Use `polars` for all DataFrame work. Never use `pandas` unless a third-party library forces it (convert immediately with `pl.from_pandas()`).
- Use `numpy` for array-level numerical work. Avoid Python-level loops over array data.
- Persist data with Parquet (`polars` native) or Arrow IPC. Avoid CSV for anything beyond small inputs.

### Computation
- I/O-bound tasks: `asyncio`.
- CPU-bound tasks: `multiprocessing` or `concurrent.futures.ProcessPoolExecutor`.
- Numerical hot paths: annotate with `@numba.njit(cache=True)` and place in the package's `_kernels.py`. Prefer `@numba.njit(parallel=True)` + `numba.prange` over manual multiprocessing for array loops.
- Prefer algorithms with O(log n) or better complexity. Avoid O(n²) loops on market data.
- Use `math` and `cmath` built-ins for scalar math — they are faster than `numpy` equivalents for scalars.

### Classes and Data Structures
- Use `@dataclass(slots=True)` for all data-holding classes. This reduces memory footprint and speeds up attribute access.
- Use `__slots__` explicitly on non-dataclass classes that are instantiated frequently. (Exception: base classes intended for subclassing with arbitrary attributes — e.g. `StrategyBase` — deliberately omit slots.)

---

## 5. Coding Style

- **Type hints everywhere.** All function signatures must be fully annotated. Pyright is in strict mode but scoped to `src/` only (`tests/`, `scripts/`, `notebooks/` are not type-checked). A clean baseline is not expected: some numba/scipy-related and pre-existing strict errors are accepted — do not chase them.
- **PEP 634 pattern matching** over `if/elif` chains for dispatch on types or structured data.
- **No comments explaining what the code does.** Only add a comment when the *why* is non-obvious (a hidden constraint, a numerical stability trick, a domain-specific invariant).
- Line length: 100 characters (`ruff` enforces this).
- Target: Python 3.11+. Use `tomllib`, `match`, `Self`, `TypeAlias`, and other 3.10–3.11 features freely.
- **Math notation is allowed to break naming rules** via `pyproject.toml` per-file ignores: `N803`/`N806` (uppercase args/vars like `S`, `K`, `T`, `F`, `P`) in `src/`; notebooks additionally tolerate `E501`, `E741`, `NPY002`, `F841`, etc. Don't "fix" these by renaming textbook variables.

---

## 6. Financial Domain Conventions

- Always be explicit about time zone when handling timestamps (`zoneinfo.ZoneInfo("UTC")` / `datetime.UTC` as default).
- Prices and returns are `float64` unless there is a specific reason to downcast.
- Keep raw market data immutable — transform into new frames rather than mutating in place.
- OHLCV frames follow the `BAR_SCHEMA` column contract defined in each package's `_types.py`; validate with the package's `validate_*` helper before computing on untrusted input.

---

## 7. Scripts, Live Trading & Credentials

- `scripts/` holds runnable CLI tools and demos. `scripts/_common.py` is the shared glue (env loading, IBKR client-id offset arithmetic, formatting) — the actual connection/quote logic lives in `qufin.trading.brokers`, not the scripts.
- The `ibkr_*.py` tools each connect on a distinct `IBKR_CLIENT_ID + offset` so read-only tools never collide with the order-placing broker connection.
- **Live order workflow:** `scripts/ibkr_order.py` defaults to a dry-run stage; the live order is sent only when `--live` is passed. Treat anything that places real orders as outward-facing — confirm before running `--live`.
- Credentials load from a gitignored project-root `.env` (template in `.env.example`: Alpaca keys, IBKR host/port/client-id). `tests/trading/conftest.py` auto-loads it so credential-gated live tests (e.g. `test_alpaca_paper_live.py`) pick up keys; those tests `skipif` the keys are absent, so they no-op in CI.

---

## 8. Notebooks

`notebooks/` is a numbered curriculum (`01_…` foundations → `27_gamma_exposure`) that doubles as tutorials for the matching `src/qufin/` packages. They are prototyping surfaces: lint rules are relaxed there (see §5) and they are not type-checked. When promoting notebook code into `src/`, tighten it to the strict-mode / fully-annotated standard.
