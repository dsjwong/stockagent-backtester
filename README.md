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
- `engine/` — event-driven daily loop, fills at next bar's open, 10 bps commission + 5 bps slippage — M2
- `strategies/` — buy-and-hold, mean reversion, trend following, LLM agents with (prompt -> decision) cache — M3/M4
- `metrics/` — CAGR, Sharpe, Sortino, max drawdown, turnover — M2

**Status:** M1 (data layer). See `docs/decisions.md` and `docs/stockagent-reuse-assessment.md`.

**Run:** `python3 -m venv .venv && .venv/bin/pip install -e . pytest` · refresh data with
`.venv/bin/python -m sab.data.download [--force]` (snapshots are committed, so this is optional) ·
`.venv/bin/pytest -q`.
