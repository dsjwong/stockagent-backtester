# StockAgent Backtester

**What:** a small, honest backtesting harness that replays real daily US-equity data through
LLM trading agents (ported from [StockAgent](https://arxiv.org/abs/2407.18957)), classical
strategies (mean reversion, trend following) and buy-and-hold, with transaction costs and
strict no-lookahead discipline.

**Why:** StockAgent trades on synthetic, session-based prices. Nobody knows whether its LLM
agents beat a moving average on real data after costs. This repo answers that, and the answer is
allowed to be "they don't".

**Architecture** (`src/sab/`):
- `data/` — fixed 15-name US large-cap universe + SPY, yfinance snapshots to parquet, offline loader (never hits network) — **done (M1)**
- `engine/` — event-driven daily loop: decide at close[t], fill at open[t+1] (never same-bar), integer
  shares, no shorting/leverage, 10 bps commission + 5 bps slippage per side; `Context` gives strategies
  an `iloc[:t+1]` view only — **done (M2)**
- `metrics/` — CAGR, vol, Sharpe/Sortino (rf = 0), max drawdown + duration, Calmar, turnover, exposure,
  win rate, costs; `summary_table()` — **done (M2)**
- `strategies/` — buy-and-hold benchmarks (EW-15 monthly rebalance, EW-15 hold, SPY reference) — done (M2);
  mean reversion, trend following — M3; LLM agents with (prompt -> decision) cache — M4

**Status:** M2 done (engine + metrics + benchmark; 32 tests incl. lookahead-perturbation, no-same-bar-fill,
exact round-trip cost, cash >= 0 / exposure <= 1, determinism). Next: M3 classical strategies.
See `docs/decisions.md` (esp. #10-#17 for engine rules) and `reports/M2-benchmark.md`.

**Run:** `python3 -m venv .venv && .venv/bin/pip install -e . pytest` ·
`.venv/bin/pytest -q` · benchmark: `.venv/bin/python scripts/run_benchmark.py` (offline; writes
`reports/M2-benchmark.md` + `reports/fig/equity_full.png`) · refresh data with
`.venv/bin/python -m sab.data.download [--force]` (snapshots are committed, so this is optional).

**Headline (M2 benchmark, with costs, rf = 0, 2018-01-02 .. 2026-08-14, universe picked with hindsight):**

| strategy | CAGR | Vol | Sharpe | MaxDD | Turnover/yr | 2021 CAGR | 2022 CAGR |
|---|---|---|---|---|---|---|---|
| EW-15 rebal (monthly) — official benchmark | 19.8% | 18.1% | 1.09 | -31.2% | 0.35 | 42.3% | -9.7% |
| EW-15 hold (no rebal) | 24.0% | 22.3% | 1.08 | -29.8% | 0.06 | 42.3% | -6.6% |
| SPY (reference) | 14.8% | 19.1% | 0.82 | -33.7% | 0.06 | 30.7% | -19.2% |

Costs cost the monthly-rebalanced portfolio ~$18.7k on $1M over 8.6 years (371% vs 376% total
return) — negligible here, decisive later for anything that trades daily. Full tables (with/without
costs, 2021 bull, 2022 drawdown) and the EW-hold sanity check are in `reports/M2-benchmark.md`.

![equity curves](reports/fig/equity_full.png)

**Minimal strategy API:**
```python
from sab.engine import Order, Context, BacktestConfig, load_market, run_backtest

class MyStrategy:
    name = "demo"
    def on_bar(self, ctx: Context) -> list[Order]:      # ctx.prices(lookback) ends at the current bar
        px = ctx.prices(20)                              # nothing after ctx.date is reachable
        return [Order.target(t, 1 / len(ctx.tickers)) for t in ctx.tickers]

res = run_backtest(MyStrategy(), load_market(), BacktestConfig(rebalance="monthly"))
res.equity, res.weights, res.trades                    # marked at close; fills at next open
```
