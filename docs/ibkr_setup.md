# Interactive Brokers — paper-trading + data setup

This is the end-to-end walkthrough for wiring an IBKR paper account into
qufin. The trading code (`qufin.trading.brokers.IBKRBroker`) and the
historical-data vendor (`qufin.data.vendors.IBKRHistoricalOHLC`) are
already in the repo; what you need to do once is (a) get a paper account
funded, (b) install and configure **IB Gateway**, and (c) point qufin at
the local socket via `.env`.

---

## 1. IBKR account: enable paper trading

You already have an IBKR account. To get a paper account on top of it:

1. Log in to <https://www.interactivebrokers.com> with your live credentials.
2. **Settings → Account Settings → Configuration → Paper Trading Account**
   → *Yes, I want a paper account*. IBKR creates a parallel paper login
   (typically your live username with a `dp` suffix; you choose the
   password).
3. The paper account is funded with **$1,000,000 virtual USD** by default,
   refreshed on request.
4. Paper accounts get free **delayed** market data for US stocks and ETFs
   with no extra subscription. Real-time data needs a subscription on the
   live account that propagates to paper.

The paper login is what you'll use everywhere below — never the live
login while you're developing.

---

## 2. Install IB Gateway (recommended over TWS)

**IB Gateway** is the headless socket-only client. It exposes the same
API as TWS but with no charting GUI, lower memory, and a more stable
auto-reconnect. Use TWS only if you also want manual charting.

1. Download **IB Gateway — Latest** from
   <https://www.interactivebrokers.com/en/trading/ibgateway-latest.php>.
   Pick the *Stable* build, not Beta.
2. Install with the defaults.
3. Launch it. On the login screen:
   - **Username / Password**: your *paper* credentials.
   - **Trading Mode**: select **Paper Trading** (not Live).
   - Click **Login**.

After login you should see a small status window with a green "API" dot.
If it's red, the API is disabled — see step 3.

---

## 3. Enable the API and pin the socket port

In IB Gateway: **Configure** (gear icon) → **Settings** → **API**:

| Setting                                    | Value                                  |
| ------------------------------------------ | -------------------------------------- |
| Enable ActiveX and Socket Clients          | ☑ checked                              |
| Socket port                                | `4002`  (paper) — `4001` for live      |
| Master API client ID                       | leave blank                            |
| Read-Only API                              | ☐ unchecked (you want to place orders) |
| Allow connections from localhost only      | ☑ checked                              |
| Bypass Order Precautions for API Orders    | ☑ checked (silences the popup)         |
| Trusted IPs                                | add `127.0.0.1`                        |

Click **OK** and **Apply**. The status dot should turn green.

> **Port cheatsheet** — these are the four ports the API uses:
>
> - `7497` = TWS paper
> - `7496` = TWS live
> - `4002` = Gateway paper
> - `4001` = Gateway live
>
> qufin defaults to `7497` (TWS paper). If you're using Gateway paper,
> set `IBKR_PORT=4002` in `.env`.

---

## 4. Configure auto-restart (optional but useful)

IB Gateway logs you out daily at ~23:45 server time. To avoid manual
re-login during development:

**Configure → Settings → Lock and Exit** → **Auto restart** → enable,
pick a time outside US market hours (e.g. 04:00 local). Once during
setup you'll need to log in manually and tick *"never deactivate
automatically"* in the Account Management portal.

---

## 5. Wire credentials into qufin

The repo already ships `.env.example` with the IBKR variables. Make sure
your real `.env` (gitignored) contains:

```bash
# Interactive Brokers — TWS or IB Gateway must be running locally
IBKR_HOST=127.0.0.1
IBKR_PORT=4002       # 4002 = Gateway paper, 7497 = TWS paper
IBKR_CLIENT_ID=1     # any int; must be unique per connected client
```

The smoke-test script and any scheduled job loads this file via a small
inline parser (same pattern as `scripts/check_alpaca_connection.py`).

---

## 6. Install the optional `ib_async` dependency

`ib-async` is declared in the `trading-live` dependency group of
`pyproject.toml`, which is not installed by `uv sync` by default. Pull
it in once:

```bash
uv sync --group trading-live
```

Verify:

```bash
uv run python -c "import ib_async; print(ib_async.__version__)"
```

---

## 7. Run the smoke test

With IB Gateway logged in to the **paper** account and the API enabled:

```bash
uv run python scripts/check_ibkr_connection.py
```

You should see:

```
IBKR paper connection OK
  cash         = $1,000,000.00
  equity       = $1,000,000.00
  buying_power = $4,000,000.00
  positions    = 0 open
Historical-data probe (SPY, 5d 1h bars, delayed):
  bars         = 35
  first        = 2026-05-17 14:30:00+00:00  close=...
  last         = 2026-05-23 20:00:00+00:00  close=...
```

If you see `TimeoutError` or `ConnectionRefusedError`, Gateway is not
running, the API is not enabled, or `IBKR_PORT` is wrong (see step 3).

---

## 8. Fetching data from a notebook or script

Historical bars use the same `OHLCSource` protocol as the Alpaca and
yfinance vendors, so they're a drop-in replacement anywhere the
data pipeline accepts a source:

```python
from datetime import datetime, UTC
from qufin.data.vendors import IBKRHistoricalOHLC

src = IBKRHistoricalOHLC(port=4002)  # Gateway paper
bars = src.fetch(
    "SPY",
    start=datetime(2026, 1, 1, tzinfo=UTC),
    end=datetime(2026, 5, 24, tzinfo=UTC),
    interval="1d",
)
print(bars.data.head())
```

For real-time bars during a paper-trade run, use the broker's
`stream_bars` (already implemented in `IBKRBroker._bar_stream`).

---

## 9. Placing a paper order

The broker surface is identical to `AlpacaBroker`, so any strategy that
runs against Alpaca paper runs against IBKR paper without changes:

```python
import asyncio
from qufin.trading.brokers import IBKRBroker
from qufin.trading._types import Order, OrderType, TimeInForce

async def main() -> None:
    broker = IBKRBroker(port=4002, client_id=2)
    await broker.connect()
    order_id = await broker.place_order(
        Order(
            asset="SPY",
            qty=1.0,
            order_type=OrderType.MARKET,
            tif=TimeInForce.DAY,
            limit_price=None,
            stop_price=None,
        )
    )
    print(f"placed: {order_id}")
    await broker.disconnect()

asyncio.run(main())
```

Use a **different `client_id`** for every concurrently connected
process — qufin's smoke test uses `1` by default, scheduled jobs should
use `2`, `3`, etc.

---

## 10. Common pitfalls

- **`API connection broken`** mid-run → Gateway logged itself out
  overnight. Set up the auto-restart from step 4.
- **`Historical Market Data Service error message: No market data
  permissions`** → the symbol needs a data subscription, or you're
  asking for live data on a delayed-only paper account. Set
  `what_to_show="TRADES"` with delayed data, or upgrade the
  subscription.
- **`Pacing violation`** → you're hitting IBKR's historical-data rate
  limits (≈60 requests / 10 minutes for small bars). Add a `await
  asyncio.sleep(...)` between batch calls or widen the bar size.
- **Orders rejected with `The contract is not available for
  trading`** → option contracts need explicit `qualifyContracts`,
  which the broker already does in `place_order`. If you wrote a custom
  flow that bypasses that, call it yourself.

---

## 11. Demo: train a strategy and paper-trade it end-to-end

Two scripts ship in `scripts/` that exercise the full path — data fetch,
training, backtest, live paper order. The underlying strategy is
intentionally simple (EMA fast/slow crossover with an ATR trailing stop,
`src/qufin/strategies/ema_cross_atr.py`) so the focus stays on the
plumbing.

### Step 1 — Train

```bash
uv run python scripts/train_ibkr_demo.py --symbol AAPL
```

This pulls ~3 years of daily AAPL bars from IBKR, grid-searches over
`fast_window × slow_window × atr_window × atr_mult` on the first 80% of
bars, reports out-of-sample Sharpe on the remaining 20%, and writes the
winning params to `artifacts/ema_cross_atr_aapl.json`.

> **EU PRIIPs note** — if your IBKR account is registered in the EU, US
> ETFs like SPY / QQQ are blocked at order-submission time (Error 201:
> "No Trading Permission, Customer Ineligible … This product does not
> have a KID"). Individual US stocks (AAPL, MSFT, …) and UCITS ETFs
> (CSPX, VWRA, …) are not subject to this, so the demo defaults to a
> single-stock example.

Override defaults with flags:

```bash
uv run python scripts/train_ibkr_demo.py --symbol MSFT --years 5 --train-frac 0.7
```

### Step 2 — Decide (dry-run)

```bash
uv run python scripts/run_ibkr_demo.py --params artifacts/ema_cross_atr_aapl.json
```

Loads the trained params, pulls the latest 400 days of bars, replays
them through the strategy to compute the *current* desired weight,
queries your paper account for the held position, and **prints** the
order it *would* have placed. Nothing is submitted.

Sample output:

```
Loaded params for SPY (trained 2026-05-25T…)
  in-sample Sharpe   = 0.842
  out-of-sample      = 0.611  (4 trades)
  MODE = DRY-RUN (no orders will be submitted)

--- cycle @ 2026-05-25T14:32:01+00:00 ---
  pulling 400d of daily SPY bars from IBKR…
  last close   = 542.18
  fast EMA     = 540.07
  slow EMA     = 532.94
  ATR          = 3.81
  trail stop   = 532.76  (highest close 544.19)
  target weight= 1.00  (LONG)
  account eq.  = $1,000,000.00  cash=$1,000,000.00
  current pos  = 0 sh  (flat)
  target pos   = 18 sh  (notional $9,759)
  → BUY 18 SPY @ MARKET (delta = +18 sh)
  (dry-run; pass --live to actually submit)
```

### Step 3 — Submit paper orders

When you're happy with the decision, add `--live`:

```bash
uv run python scripts/run_ibkr_demo.py --params artifacts/ema_cross_atr_aapl.json --live
```

Optional safety knobs:

* `--max-position-usd 5000` — hard cap on dollar notional (default $10k).
* `--interval 300` — loop every 5 min instead of running once.
* `--history-days 800` — widen the warm-up window if you trained on a
  longer history.

### What it doesn't do

* Short-side trades — long-or-flat only.
* Multiple symbols — by design; widen `make_strategy` and the runner if
  you want a portfolio.
* Intraday — daily bars only; for intraday wire `current_state` into a
  loop reading `IBKRBroker.stream_bars`.

---

## 12. Demo: options version (long-call trend follower)

A second pair of scripts in `scripts/` runs the same EMA-cross signal
but trades long calls instead of shares. The strategy lives in
[`src/qufin/strategies/ema_cross_long_call.py`](../src/qufin/strategies/ema_cross_long_call.py).

### How it differs from the equity demo

| | Equity demo | Options demo |
|---|---|---|
| Asset | Shares | One long ATM-ish call per cycle |
| Entry | EMA cross-up | EMA cross-up |
| Exit | EMA cross-down or ATR trailing stop | EMA cross-down *or* DTE ≤ `exit_dte` |
| Backtest pricing | Underlying close | Black-Scholes (`FlatIVMarkProvider`) at the underlying's next-bar open |
| Backtest IV | n/a | Realised vol of underlying log returns (annualised, clamped to [10%, 80%]) |
| Live strike/expiry | n/a | Closest listed pair to `spot × strike_moneyness` and `today + target_dte` |
| Order surface | `IBKRBroker.place_order(asset=symbol_str, …)` | `…(asset=OptionContract, …)` |

### Step 1 — Train

```bash
uv run python scripts/train_ibkr_options_demo.py --symbol AAPL
```

Pulls 3y of AAPL daily bars, estimates the flat-IV assumption from
realised vol, grid-searches `fast/slow EMA × strike_moneyness ×
target_dte × exit_dte` on the in-sample slice, then replays the best
combination out-of-sample. Writes
`artifacts/ema_cross_long_call_aapl.json`.

Optional overrides:

```bash
uv run python scripts/train_ibkr_options_demo.py \
    --symbol MSFT --years 5 --iv 0.30
```

### Step 2 — Decide (dry-run)

```bash
uv run python scripts/run_ibkr_options_demo.py \
    --params artifacts/ema_cross_long_call_aapl.json
```

Sample output:

```
Loaded params for AAPL (trained 2026-05-25T…)
  in-sample Sharpe   = 0.71
  out-of-sample      = 1.20  (3 trades)
  MODE = DRY-RUN (no orders will be submitted)

--- cycle @ 2026-05-25T16:00:00+00:00 ---
  pulling 400d of daily AAPL bars from IBKR…
  last close       = 308.82
  fast EMA / slow  = 295.40 / 271.10
  desire           = LONG CALL
  flat-IV (backtest)= 0.282
  suggested target = AAPL 2026-06-24 309.0C
  account eq.      = $1,000,000.00  cash=$1,000,000.00
  current calls    = 0
  → BUY  1 AAPL 2026-06-26 310.0C  (target 2026-06-24 309.0, DTE=32)
  (dry-run; pass --live to actually submit)
```

The runner queries IBKR's option chain (`reqSecDefOptParams`) and picks
the listed (expiry, strike) closest to the strategy's synthetic target.
Note `2026-06-26` (Friday) vs the synthetic `2026-06-24` (Wednesday) —
that's the closest standard monthly.

### Step 3 — Submit paper orders

```bash
uv run python scripts/run_ibkr_options_demo.py \
    --params artifacts/ema_cross_long_call_aapl.json --live
```

Safety knobs:

* `--max-contracts 1` — hard cap on contracts opened per cycle.
* `--interval 300` — loop every N seconds.

### Caveats specific to the options path

* **PRIIPs again** — even individual US stocks are fine for the equity
  demo on EU paper accounts, but a small subset of US-listed options
  may still be blocked. If you get Error 201 on `AAPL`, try a more
  liquid name (`SPY` options, ironically, are often *allowed* for EU
  retail accounts even when the underlying ETF isn't — the rule
  attaches to the packaged product, not the option).
* **Flat IV underprices the wings** — the backtest's constant-IV
  assumption is fine for the demo's purpose (showing the plumbing
  works) but understates the cost of OTM strikes. Treat the reported
  Sharpe as directional, not absolute.
* **Strike granularity** — the synthetic backtest rounds to $1 strikes
  via `strike_step`; live, we snap to the closest listed strike, which
  may be $5 or $10 apart on high-priced underlyings.
* **No early exit on adverse moves** — there's no stop-loss on premium.
  Add one in `EmaCrossLongCallStrategy.on_bar` if you want it.
