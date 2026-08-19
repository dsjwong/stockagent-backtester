# StockAgent Backtester

Real-data, no-lookahead backtesting of LLM trading agents against classical strategies and buy-and-hold, with
transaction costs. The LLM agent comes from [StockAgent](https://arxiv.org/abs/2407.18957) (a multi-agent
LLM market simulation on synthetic prices); this repo replays it on 15 real US large-caps (2019-2026), caches every
prompt so a replay costs $0, anonymises tickers and dates so the model cannot trade on what it remembers, and
reports the same metrics for the LLM, mean reversion, trend following, weighted trend, momentum and equal-weight
buy-and-hold in a 2021 bull year and the 2022 drawdown. The answer is allowed to be, and mostly is, "the LLM does
not beat equal weight on risk-adjusted terms after costs" -- the anonymised Gemini agent is a high-turnover momentum
chaser that wins 2022 for the same reason every momentum rule does (energy), trails in 2021, and has the worst
drawdowns of anything but mean reversion; naming the tickers changes its behaviour in ways that look like
hindsight, which is why that line is reported but not trusted.

## Headline results (with costs 10 bps commission + 5 bps slippage per side, rf = 0, defaults untuned)

| strategy | rebal | 2021 CAGR | 2021 Sharpe | 2021 MaxDD | 2022 CAGR | 2022 Sharpe | 2022 MaxDD | turnover/yr | full 2019-26 CAGR / Sharpe |
|---|---|---|---|---|---|---|---|---|---|
| **LLM gemini-3.5-flash-lite, anon** (weekly, T=0) | weekly | 41.5% | 2.29 | -7.4% | **1.6%** | 0.18 | -21.6% | 8.6-9.3 | incomplete: 253/397 decisions cached (resume, see reports/M4-llm-results.md) |
| LLM gemini-3.5-flash-lite, **real** names/dates | weekly | 32.9% | 2.37 | -6.0% | 12.0% | 0.70 | -15.5% | 4.4-4.5 | -- |
| MeanRev (20d z < -1, EW) | weekly | 19.3% | 1.18 | -6.5% | -18.2% | -0.56 | -27.4% | 27.5 | 2.5% / 0.22 |
| Trend (SMA50>SMA200, EW among trenders) | weekly | 40.8% | 2.81 | -4.1% | -3.3% | -0.08 | -17.1% | 2.6 | 21.6% / 1.18 |
| Trend (SMA50>SMA200, 1/N slots) | weekly | 38.2% | 2.79 | -4.1% | -4.3% | -0.38 | -8.8% | 1.2 | 14.6% / 1.13 |
| WeightedTrend (1/3/6/12m blend, 25% cap) | monthly | 40.4% | 2.18 | -5.9% | 3.0% | 0.24 | -14.2% | 2.5 | 26.8% / 1.21 |
| MomTop5 (12-1m) | monthly | 40.9% | 1.72 | -11.5% | 1.7% | 0.19 | -16.3% | 2.4 | 25.3% / 1.07 |
| **EW-15 rebal (official benchmark)** | monthly | 42.3% | 2.97 | -4.1% | -9.7% | -0.35 | -19.6% | 0.4-0.8 | 22.7% / 1.23 |
| EW-15 hold | -- | 42.3% | 2.83 | -4.2% | -6.6% | -0.22 | -18.2% | 0.1 | 31.4% / 1.22 |
| SPY (reference, not tradeable) | -- | 30.7% | 2.14 | -5.1% | -19.2% | -0.75 | -24.8% | 0.1 | 17.9% / 0.95 |

LLM without costs: 45.5% (2021) / 4.3% (2022) anon, 34.7% / 13.5% real -- the same logged decisions replayed at zero
cost, so the 2.6-4.0 pp gap is pure turnover. Consistency (same prompt, 8 fresh calls at T=0.7 and T=0 on 2022-06-13):
anonymised, the same 4 names every time with weights moving by 0.1; real names, 100% cash every time. Full tables,
figures, rationales, repair/fallback counts (0 parse failures in 208 decisions) and the interpretation:
[`reports/M4-llm-results.md`](reports/M4-llm-results.md) (LLM) and [`reports/M3-results.md`](reports/M3-results.md)
(classical, 4 periods x with/without costs, cost sweep 0/15/30/60 bps).

![LLM vs benchmarks, 2022](reports/fig/llm_2022.png)
![classical strategies, 2019-2026](reports/fig/equity_classical_full.png)

## Methodology

- **Data**: 15 US large-caps (AAPL MSFT NVDA AMZN GOOGL JPM BAC XOM CVX JNJ PFE UNH PG KO WMT) + SPY, daily adjusted
  OHLCV from yfinance, 2018-01-02 .. 2026-08-14, snapshotted to parquet and committed; backtests never touch the network.
- **Engine**: event-driven daily loop. Decisions at close[t], **fills at open[t+1]** (never same-bar), integer shares, no
  shorting/leverage, 10 bps commission + 5 bps slippage per side (30 bps round trip); every table also exists at zero cost.
- **Lookahead by construction and by test**: strategies get a `Context` holding `iloc[:t+1]` views only; perturbation tests
  multiply all prices after T* by 1.5 and assert equity/fills through T* are identical (and that later decisions do change);
  the LLM prompt builder is audited by formatting every post-decision price/date the way the prompt does and asserting none
  appears, on real data, plus the same perturbation harness on the agent with a hash-mock model.
- **LLM agent**: one call per weekly decision; prompt = rules, 60-bar close table, 1w/1m/3m/6m trailing returns, current
  portfolio, strict JSON output spec (`{"weights": {...}, "rationale": "<= 40 words"}`); tolerant parser with every repair
  counted, one retry then hold-previous-weights; per-decision log (prompt hash, tokens, latency, exposure, rationale, repairs).
- **Two prompt modes**: `real` (symbols, dates, adjusted closes) and `anon` (`Stock 01..15` via a seeded shuffle, `Day t`,
  prices rebased to 100) -- the anon numbers are "what the model does with the data", the real numbers an upper bound
  contaminated by whatever the model knows about 2021/2022.
- **Cache**: every (provider, model, temperature, seed, prompt) -> response is stored under `data/llm_cache/` and committed;
  replaying any result here makes zero API calls (tested). Real runs are paced to the free-tier rate limit, back off on 429,
  and stop cleanly on a daily-quota 429; re-running the same command resumes from the cache.
- **Metrics**: CAGR, vol, Sharpe/Sortino (rf = 0, stated everywhere), max drawdown + duration, Calmar, annualised turnover,
  exposure, win rate, costs. Benchmark = EW-15 monthly rebalance; EW-hold and SPY alongside.
- **No tuning**: every strategy runs on textbook defaults; nothing was fitted to any reporting period (decisions.md #19).

## Limitations (read before quoting any number)

- **Survivorship / hindsight universe**: the 15 names were chosen in 2026; EW-15 hold's 31% CAGR is mostly NVDA. Every
  absolute number is inflated; relative comparisons on the same universe are the point.
- One market, daily bars, 15 names, two regime years for the LLM (52 decisions each), one model, one seed: differences of a
  few percentage points are noise, and "beat EW in 2022" is what every momentum rule did.
- **Free-tier model choice**: `gemini-3.5-flash-lite` (the planned `gemini-2.5-flash` is gone for new users; flash models are
  capped at 20 requests/day on the free tier), thinking set to minimal. A stronger or differently-prompted model may behave
  differently; the prompt hands the model trailing returns, which makes momentum the path of least resistance.
- rf = 0 in Sharpe/Sortino (2018-2026 T-bills ~2.3% would lower every Sharpe by ~0.1); costs are a flat bps model, no market
  impact; fills at the open with no volume check; integer shares on $1M so weights are approximate.
- Real-mode prompts cannot be audited for leakage through the model's weights (only through the prompt); that is what the anon
  mode is for, and the real-vs-anon gap in `reports/M4-llm-results.md` is the evidence it matters.

## Reproduce

```bash
python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'                 # Python 3.12+, all deps offline-capable
.venv/bin/pytest -q                                                        # 107 tests: fills, lookahead, prompt audit, cache, parser, adapter
.venv/bin/python scripts/run_classical.py                                  # M3 tables + figures from the committed snapshot (~1 min)
.venv/bin/python scripts/run_llm.py --provider gemini --model gemini-3.5-flash-lite --mode anon --period 2022   # 0 API calls: served from the committed cache
.venv/bin/python scripts/report_llm.py                                     # rebuilds reports/M4-llm-results.md + reports/fig/llm_*.png
```

New model or period: put `GOOGLE_API_KEY` (or `ANTHROPIC_API_KEY` / `OPENAI_API_KEY`; `pip install -e '.[llm]'` for those SDKs)
in `.env`, then e.g. `scripts/run_llm.py --provider gemini --model gemini-3.5-flash-lite --mode anon --period full --verbose`
(exits 2 with a resume hint if the daily quota runs out; re-run the same command later). `--estimate-tokens` prices a run
before any call; `--consistency 2022-06-13 --n 5 --ctemp 0.7` repeats one prompt. Mock providers (`--provider mock`,
`mock_ew`, `mock_random`) exercise the whole pipeline offline. Refresh data with `python -m sab.data.download --force`.

## Layout

```
src/sab/data/        universe, yfinance/stooq download, offline parquet loader
src/sab/engine/      MarketData, Context (point-in-time view), broker (next-open fills, costs), run_backtest, BacktestResult (save/load)
src/sab/metrics/     CAGR/Sharpe/Sortino/MaxDD/turnover/... + summary_table
src/sab/strategies/  EqualWeight, SpyReference, MeanReversion, TrendFollowing, WeightedTrend, MomentumTopK
src/sab/llm/         prompt builder + no-future audit, disk cache, providers (gemini REST, anthropic, openai, ollama, 3 mocks), parser, LLMAgent, consistency_check
scripts/             run_benchmark.py (M2), run_classical.py (M3), run_llm.py (M4), report_llm.py (M5)
reports/             M2-benchmark.md, M3-results.md, M4-llm-results.md, fig/, runs/<period>/<cost>/<strategy>/ (equity/weights/trades/decisions parquet + config)
data/                snapshots (committed), llm_cache/ (committed prompt -> response)
docs/decisions.md    numbered design decisions (#1-#29) with the reasoning; docs/stockagent-reuse-assessment.md
tests/               107 offline tests
```

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
