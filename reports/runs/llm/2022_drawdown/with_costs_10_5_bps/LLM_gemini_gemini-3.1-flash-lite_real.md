# LLM(gemini/gemini-3.1-flash-lite, real) on 2022 (drawdown)

costs 10+5 bps/side, rf=0

| strategy (rf=0) | CAGR | TotalRet | Vol | Sharpe | Sortino | MaxDD | DD_days | Calmar | Turnover | Exposure | WinRate | Trades | Costs | Costs_bps |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LLM(gemini/gemini-3.1-flash-lite, real) | 8.3% | 8.2% | 20.1% | 0.49 | 0.71 | -13.8% | 168 | 0.60 | 9.88 | 94.9% | 65.8% | 362 | 31,012 | 310 |
| EW-rebal (monthly) | -9.7% | -9.6% | 22.1% | -0.35 | -0.49 | -19.6% | 191 | -0.50 | 0.79 | 99.1% | 45.0% | 178 | 2,299 | 23 |
| EW-hold | -6.6% | -6.5% | 20.8% | -0.22 | -0.31 | -18.2% | 191 | -0.36 | 0.50 | 99.5% |  | 15 | 1,497 | 15 |
| SPY (reference) | -19.2% | -19.0% | 24.3% | -0.75 | -1.04 | -24.8% | 250 | -0.77 | 0.50 | 99.6% |  | 1 | 1,498 | 15 |
| Trend(SMA50>SMA200) | -3.3% | -3.3% | 19.3% | -0.08 | -0.11 | -17.1% | 191 | -0.19 | 3.63 | 99.1% | 59.2% | 402 | 10,315 | 103 |
| Trend(SMA50>SMA200), 1/N slots | -4.3% | -4.3% | 10.3% | -0.38 | -0.52 | -8.8% | 191 | -0.49 | 1.52 | 50.6% | 61.6% | 405 | 4,408 | 44 |
| WeightedTrend(21/63/126/252) | 3.0% | 2.9% | 20.9% | 0.24 | 0.34 | -14.2% | 155 | 0.21 | 2.69 | 97.1% | 38.7% | 125 | 7,909 | 79 |

```json
{
 "period": "2022 (drawdown)",
 "costs": {
  "commission_bps": 10.0,
  "slippage_bps": 5.0
 },
 "agent": {
  "provider": "gemini",
  "model": "gemini-3.1-flash-lite",
  "mode": "real",
  "lookback": 60,
  "temperature": 0.0,
  "seed": 0,
  "max_tokens": 400,
  "rebalance": "weekly",
  "warmup": 127,
  "max_retries": 1,
  "anon_mapping": null
 },
 "n_decisions": 52,
 "n_model_calls": 52,
 "n_provider_calls_this_run": 52,
 "cache_hits": 0,
 "prompt_tokens": 174529,
 "completion_tokens": 5635,
 "retries": 0,
 "fallbacks": 0,
 "repairs": 458,
 "mean_exposure": 0.9567307692307693
}
```
