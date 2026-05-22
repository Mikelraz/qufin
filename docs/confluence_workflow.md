# Workflow: training, backtesting, and deploying the WHCG strategy

The Wyckoff-Hull Confluence Strategy with GEX-Aware Defense (WHCG) lives in
`qufin.strategies.confluence`. This document is the operational runbook —
what to run, in what order, and what to check at each stage before moving
on.

---

## Stage 0 — Environment & sanity check

One-time setup, plus a recurring "is the pipeline healthy" gate.

```bash
uv sync --all-groups                                    # install deps incl. dev group
ruff check --fix . && ruff format .                     # lint + autoformat
pyright src/qufin/strategies/confluence                 # strict-mode type check
pytest -n auto tests/strategies/test_confluence_*.py    # 14 unit + 2 integration
```

**Gate before moving on:** all checks pass. The integration test in
particular exercises the whole pipeline (regime → signals → GEX → sizing
→ engine) on synthetic data — if it breaks, the next stages produce
noise.

`.env` in repo root needs:

```
ALPACA_API_KEY=...
ALPACA_API_SECRET=...
```

The paper-trade script reads these automatically via the in-script
`_load_env` helper.

---

## Stage 1 — "Training" the edge model

The strategy is **rule-based, not parametric-fit** — there is no
`model.fit()` call. What needs training is the per-symbol Kelly **edge**
(hit-rate `p` and win/loss ratio `b`). That is bootstrapped in two ways.

### 1a. Cold start (no historical trade ledger yet)

`ConfluenceParams.default_edge_p = 0.50` and `default_edge_b = 1.0` give
every symbol a neutral-Kelly weight equal to `0.5 × 0.5 = 0.25` of the
per-name cap. **The first few months of paper trading are intentionally
undersized** until real trade outcomes accumulate.

### 1b. Warm start from a historical backtest

Run a multi-year backtest, then post-process its trade ledger to seed
per-symbol `(p, b)` priors:

```bash
uv run python scripts/confluence_backtest.py \
    SPY QQQ IWM XLK XLF XLE XLV XLI XLY XLP XLU XLB XLRE XLC \
    --start 2015-01-01 --end 2022-12-31 \
    --report-dir reports/training/
```

The trade-ledger Parquet (`reports/training/confluence_trades_*.parquet`)
gives you per-symbol realised returns. Pre-populating
`ConfluenceStrategy._trade_returns` with those before going live skips
the cold-start window. (Today the strategy reads from its in-memory
dict — if you want this warm start, expose a `prime_from_trades(df)`
method; trivial follow-up.)

### 1c. Hyperparameter tuning (optional, do not over-rely on)

The `ConfluenceParams` knobs worth a coarse grid sweep:

- `min_confluences` ∈ {2, 3, 4} — sensitivity vs precision
- `hull_fast_length, hull_slow_length` ∈ {(40, 60), (50, 60), (50, 90)}
- `chandelier_atr_mult` ∈ {2.5, 3.0, 3.5}
- `kelly_fraction` ∈ {0.25, 0.5} — half- vs quarter-Kelly

**Rule:** tune on 2015–2020, validate on 2021–2022, untouched test on
2023–2025. Anything that does not generalise across those three windows
is overfit. Walk-forward CV (Stage 2b) is the real check.

---

## Stage 2 — Backtesting

### 2a. Single-pass historical backtest

```bash
uv run python scripts/confluence_backtest.py \
    SPY QQQ IWM XLK XLF XLE XLV XLI XLY XLP XLU XLB XLRE XLC \
    --start 2018-01-01 --end 2025-12-31 \
    --report-dir reports/ \
    --log-level INFO --heartbeat 50
```

Watch the heartbeat log: equity should grow, position count should
oscillate 0–5, runtime is dominated by HMM refits (~5–10 minutes for
the full universe over 8 years).

**Outputs** (`reports/`):

- `confluence_equity_<stamp>.parquet` — full equity curve
- `confluence_trades_<stamp>.parquet` — per-fill log
- `confluence_summary_<stamp>.json` — Sharpe, Sortino, max-DD, CAGR, Calmar

**Acceptance gates** (from the plan):

- CAGR ≥ SPY buy-and-hold (or higher Sharpe with lower MDD)
- Max DD < 0.7 × SPY buy-and-hold MDD
- Sharpe ≥ 0.8, Sortino ≥ 1.1
- Exposure 40–80% (proves cash defense engages but isn't always off)
- ≥ 80 trades (statistical power)

If any gate fails → do **not** advance to paper trading. Inspect the
trade ledger first: is the loss concentrated in one regime? One symbol?
One Wyckoff event type?

### 2b. Walk-forward cross-validation

The honesty check — same strategy, multiple disjoint test windows.

```bash
uv run python scripts/confluence_walkforward.py \
    SPY QQQ IWM XLK XLF XLE XLV XLI XLY XLP XLU XLB XLRE XLC \
    --start 2018-01-01 --end 2025-12-31 \
    --train-years 2 --test-months 6 --step-months 6
```

Per-fold INFO lines report Sharpe / CAGR / MDD. Acceptance gates:

- Mean out-of-sample Sharpe ≥ 0.5 × in-sample Sharpe (degradation cap)
- No fold with negative CAGR

A wide std of per-fold Sharpe is a red flag — it means the strategy
works in some regimes and not others. Either accept it and add a regime
gate, or go back to Stage 1c.

### 2c. Trade attribution (notebook)

Open `notebooks/confluence_strategy_research.ipynb` (you'll create it
from `reports/confluence_trades_<stamp>.parquet`) and check:

- P&L by triggering Wyckoff event (Spring vs SOS vs harmony)
- P&L by exit reason (Hull flip vs swing stop vs chandelier vs cash defense)
- Per-symbol P&L — is one ETF carrying everything?

**No single event type / no single symbol should dominate.** If one
does, the strategy is brittle.

---

## Stage 3 — Pre-deployment validation

Three things must be true before money goes to Alpaca:

1. **Backtest gates pass** (Stage 2a).
2. **Walk-forward degradation is bounded** (Stage 2b).
3. **Attribution is diversified** (Stage 2c).

If you skipped any, do not start paper trading. The signals will look
fine for a week and then collapse on the first regime change.

---

## Stage 4 — Paper trading

### 4a. Dry run first

```bash
uv run python scripts/confluence_paper_trade.py --dry-run --log-level INFO
```

This:

1. Hits Alpaca for the trailing 540 daily bars of each symbol.
2. Optionally fetches SPY's option chain for GEX flags.
3. Rebuilds the strategy state by replaying the engine over those bars.
4. Prints target weights + writes `reports/paper/signals_<stamp>.json`.
5. Submits **no orders**.

Verify the proposed weights are sane (size within `max_weight_per_name`,
no name shows up that shouldn't). If GEX is enabled and SPY is below
zero-gamma, target weights should be ~halved.

### 4b. Arm paper trading

```bash
uv run python scripts/confluence_paper_trade.py --log-level INFO
```

Same as above but submits notional market-on-open orders via
`AlpacaBroker(paper=True)`.

### 4c. Schedule it

This is a daily-bar strategy, so:

- **Windows**: Task Scheduler → daily 22:00 ET (after close, before next-day open)
- **Linux/macOS**:
  ```cron
  0 2 * * 1-5  cd /path/to/qufin && uv run python scripts/confluence_paper_trade.py >> reports/paper/cron.log 2>&1
  ```

The logging output goes to the cron log — that's your audit trail.

### 4d. Daily monitoring (4–6 weeks)

Each morning:

1. Tail `reports/paper/cron.log` — any warnings or order failures?
2. Compare the day's `signals_<stamp>.json` to actual Alpaca positions (slippage attribution).
3. Weekly: re-run `confluence_backtest.py` over the paper window and compare realised Sharpe to backtest expectation.

**Acceptance gate for going live with real capital** (not in the plan,
but the next stage):

- Realised Sharpe within 1 std of walk-forward Sharpe
- Cash-defense exits fired correctly on at least one drawdown event
- No execution surprises (fills near expected prices)

---

## Stage 5 — Operating the deployed strategy

### 5a. Retraining cadence

- **HMM regime classifier**: already refits inside the strategy every
  `regime_refit_period = 21` bars. No external retrain needed.
- **Kelly edges**: rolling `edge_lookback_days = 252` of trade returns is
  read on every `on_bar`. Self-updating.
- **Covariance**: Ledoit-Wolf cov is recomputed every bar over the last
  `cov_lookback_days = 252` returns. Self-updating.

**The strategy retrains itself on every bar.** What you decide manually
is whether to nudge `ConfluenceParams` after enough out-of-sample data
accumulates — typically once every 6–12 months, only if walk-forward CV
on the latest data clearly justifies it.

### 5b. Universe rotation

The universe is hard-coded in the scripts (`DEFAULT_UNIVERSE`). Reviewing
it quarterly is reasonable:

- Add a sector ETF that's becoming more liquid.
- Drop one whose options are too thin for GEX (the overlay degrades silently).

### 5c. Going live (beyond the plan)

When paper-trade results meet the gate above, switch the broker
constructor in `confluence_paper_trade.py`:

```python
broker = AlpacaBroker(paper=False)
```

…and start with a small fraction of `max_weight_per_name` (e.g. 1%
instead of 5%) for the first month of live trading. Scale up only after
live behaviour matches paper.

---

## End-to-end ordering, distilled

```
1. uv sync && ruff && pyright && pytest                   gate: pipeline healthy
2. confluence_backtest.py    (2015-2022, warm-start)      gate: meets backtest criteria
3. confluence_walkforward.py (2018-2025)                  gate: OOS Sharpe ≥ 0.5× IS
4. notebook attribution                                   gate: no single source dominates
5. confluence_paper_trade.py --dry-run                    gate: sane targets
6. confluence_paper_trade.py + cron, 4-6 weeks            gate: paper Sharpe ≈ walk-forward
7. flip paper=False, small size, 1 month observation      gate: live ≈ paper
8. scale to full target size
```

Each `→` is a stop-or-go decision point. **The strategy retrains itself
on every bar; what humans decide is whether to advance to the next
stage.** That's the operational discipline that separates "I built a
strategy" from "I deployed a strategy."
