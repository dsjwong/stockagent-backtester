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
  classical: mean reversion (20d z-score), trend following (SMA50>SMA200, two sizings), weighted trend
  (1/3/6/12-month blend), 12-1 momentum top-5 — **done (M3)**; LLM agents with (prompt -> decision) cache — M4

**Status:** M3 done (classical strategies + first results table; 65 tests incl. the lookahead-perturbation
harness run over every strategy, no-same-bar-fill, exact round-trip cost, cash >= 0 / exposure <= 1, determinism,
synthetic behaviour checks). Next: M4 LLM agents. See `docs/decisions.md` (#10-#17 engine rules, #18-#21 strategy
defaults / no-tuning / period handling / cost sweep), `reports/M3-results.md`, `reports/M2-benchmark.md`.

**Run:** `python3 -m venv .venv && .venv/bin/pip install -e . pytest` ·
`.venv/bin/pytest -q` · classical results: `.venv/bin/python scripts/run_classical.py` (~1 min, offline; writes
`reports/M3-results.md`, `reports/fig/equity_classical_full.png` + per-period figures, and every run's
equity/weights/trades/config under `reports/runs/`) · M2 benchmark: `.venv/bin/python scripts/run_benchmark.py` ·
refresh data with `.venv/bin/python -m sab.data.download [--force]` (snapshots are committed, so this is optional).

**Headline (M3, with costs 10+5 bps/side, rf = 0, all runs entered 2019-01-02 with warmup history, universe picked with hindsight):**

| strategy (defaults, untuned) | rebal | CAGR | Vol | Sharpe | MaxDD | Turnover/yr | 2021 | 2022 | 2023-26 | CAGR @ 0 / 60 bps RT |
|---|---|---|---|---|---|---|---|---|---|---|
| MeanRev (20d z < -1, EW) | weekly | 2.5% | 21.9% | 0.22 | -35.9% | 27.5 | 19.3% | -18.2% | 4.1% | 11.3% / -5.6% |
| Trend (SMA50>SMA200, EW among trenders) | weekly | 21.6% | 17.9% | 1.18 | -28.3% | 2.6 | 40.8% | -3.3% | 20.5% | 22.5% / 20.6% |
| Trend (SMA50>SMA200, 1/N slots) | weekly | 14.6% | 12.8% | 1.13 | -23.7% | 1.2 | 38.2% | -4.3% | 14.0% | 15.1% / 14.2% |
| WeightedTrend (1/3/6/12m blend, 25% cap) | monthly | 26.8% | 21.6% | 1.21 | -28.6% | 2.5 | 40.4% | 3.0% | 27.3% | 27.8% / 25.9% |
| MomTop5 (12-1m) | monthly | 25.3% | 23.6% | 1.07 | -33.3% | 2.4 | 40.9% | 1.7% | 26.4% | 26.2% / 24.4% |
| **EW-15 rebal — official benchmark** | monthly | 22.7% | 18.1% | 1.23 | -31.2% | 0.4 | 42.3% | -9.7% | 25.9% | 22.9% / 22.6% |
| EW-15 hold | — | 31.4% | 24.9% | 1.22 | -30.7% | 0.1 | 42.3% | -6.6% | 34.2% | 31.4% / 31.4% |
| SPY (reference) | — | 17.9% | 19.4% | 0.95 | -33.7% | 0.1 | 30.7% | -19.2% | 23.2% | — |

Short version: nothing has a better Sharpe than equal-weight buy-and-hold on this (survivorship-biased) universe;
the momentum-weighted rules earn more CAGR by concentrating in the eventual winners, the 1/N trend filter buys
lower drawdown (-8.8% in 2022 at 51% exposure) with lower return, and 20-day mean reversion at weekly rebalance
bleeds out on costs (28x turnover: +11% CAGR free, +2.5% at 30 bps, -5.6% at 60 bps). Full tables (4 periods x
with/without costs), cost sweep and interpretation: `reports/M3-results.md`.

![equity curves](reports/fig/equity_classical_full.png)

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
