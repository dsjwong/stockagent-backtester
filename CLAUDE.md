# StockAgent Backtester — Real Data + Rigorous Backtesting

Extension of Darren's existing **StockAgent** project (multi-agent LLM stock-market simulation: portfolio state management, order matching, 4 LLM providers) — replacing synthetic prices with real historical market data and adding a proper backtesting engine, so LLM-agent strategies can be evaluated honestly against classical strategies and buy-and-hold. Owner: Darren (final-year HKUST CS + AI). Purpose: depth for quant/bank-tech interviews — this project is the natural deep-dive topic in every finance-tech conversation, and rigorous backtesting is exactly the vocabulary those interviews use.

## What "done" looks like (v1)
A results README an interviewer can read: equity curves and a metrics table (CAGR, Sharpe, Sortino, max drawdown, turnover, win rate) for LLM agents vs Mean Reversion vs Trend Following vs buy-and-hold, on real daily data across 2 distinct market regimes (e.g., 2021 bull, 2022 drawdown), **with transaction costs**, plus a one-page writeup of methodology and honest limitations. A dashboard is optional polish, not the deliverable.

## Scope guardrails
- Daily OHLCV bars only. No intraday, no options, no live trading.
- One market to start (US large-caps or HK blue chips — pick and commit; ~10–20 tickers).
- LLM cost control is a design constraint from day 1: cache every (prompt → decision) pair; replaying a backtest must cost $0 in API calls. Consider a local model via Ollama for cheap iteration (already integrated in StockAgent).

## Data
- **yfinance** (free, daily OHLCV, adjusted closes) as primary; **Stooq** as fallback. Store snapshots locally (parquet/SQLite) so runs are reproducible — never hit the network during a backtest.
- Corporate actions: use adjusted prices; document the choice.

## Backtesting engine (the actual engineering showcase)
- Event-driven daily loop: for each bar → strategies emit orders → fills at next bar's open (NO same-bar fills — that's lookahead) → portfolio accounting.
- Costs: commission (e.g., 10 bps/trade) + slippage model (e.g., 5 bps). Show results with AND without costs — the delta is itself an interesting finding.
- **Lookahead/leakage discipline is the whole game**: strategies see only data up to decision time; LLM prompts must not contain future bars (audit the prompt builder); walk-forward splits for anything with tunable parameters.
- Metrics module: CAGR, Sharpe (state the rf assumption), Sortino, max drawdown + duration, turnover, exposure. Benchmark: buy-and-hold equal-weight.
- Reuse StockAgent's portfolio/order-matching code where it fits, but the simulator's session-based pricing gets replaced by historical replay.

## Experiments (the story the results tell)
1. Classical baselines: Mean Reversion, Trend Following, Weighted Trend (port from the C++ project) on real data.
2. LLM agents (per provider, incl. a local Ollama model): same replay, prompts fed with only-past data + portfolio state. Measure decision consistency (same input twice → same action?) — a genuinely interesting finding either way.
3. Regime comparison: run everything on a bull year and a drawdown year separately.
4. (Stretch) Ensemble/committee of LLM agents voting vs single agent.

Honest framing matters more than good numbers: "LLM agents underperformed trend-following after costs, and here's why" is a GREAT interview story.

## Milestones
- **M1 (1–2 evenings)**: data layer — download, snapshot, load; 10–20 tickers, 5+ years daily.
- **M2 (2–3 evenings)**: backtest engine + metrics + buy-and-hold benchmark; unit tests for fill logic and lookahead (test that shifting future data changes nothing).
- **M3 (1–2 evenings)**: port classical strategies; produce first results table.
- **M4 (2–3 evenings)**: wire StockAgent LLM agents into replay with decision caching; run experiments.
- **M5 (1 evening)**: README with equity-curve plots, metrics table, methodology, limitations. Then a CV bullet with real numbers.

## Working notes for Claude sessions in this repo
- The existing StockAgent codebase is the starting point — locate it (ask Darren for the repo path / GitHub link) and assess what's reusable before writing anything new.
- Python 3.12+, venv in `.venv/`; plotting with matplotlib is fine. Keep API keys in `.env`, git-ignored.
- Owner's time is scarce (job application season): prefer the smallest runnable step; M2's correctness (no lookahead, tested fills) is the part interviewers probe — never trade it for speed.
- Context: job-hunt materials in `~/Desktop/Job Hunting 2026-2027/`. Quant applications go out late September 2026 — an M3 results table by then is the payoff; M4's LLM comparison can land in October.
