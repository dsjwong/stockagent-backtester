# LLM(gemini/gemini-3.5-flash-lite, real) on 2021 (bull)

costs 10+5 bps/side, rf=0

| strategy (rf=0) | CAGR | TotalRet | Vol | Sharpe | Sortino | MaxDD | DD_days | Calmar | Turnover | Exposure | WinRate | Trades | Costs | Costs_bps |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LLM(gemini/gemini-3.5-flash-lite, real) | 32.9% | 32.7% | 12.3% | 2.37 | 3.74 | -6.0% | 35 | 5.51 | 4.36 | 90.4% | 86.7% | 651 | 15,061 | 151 |
| EW-rebal (monthly) | 42.3% | 42.1% | 12.1% | 2.97 | 4.54 | -4.1% | 28 | 10.42 | 0.77 | 99.2% | 95.8% | 179 | 2,457 | 25 |
| EW-hold | 42.3% | 42.1% | 12.8% | 2.83 | 4.35 | -4.2% | 28 | 10.17 | 0.50 | 99.5% |  | 15 | 1,497 | 15 |
| SPY (reference) | 30.7% | 30.6% | 12.9% | 2.14 | 3.21 | -5.1% | 32 | 6.00 | 0.50 | 99.6% |  | 1 | 1,498 | 15 |
| Trend(SMA50>SMA200) | 40.8% | 40.6% | 12.5% | 2.81 | 4.29 | -4.1% | 28 | 9.97 | 1.61 | 99.3% | 90.5% | 728 | 5,396 | 54 |
| Trend(SMA50>SMA200), 1/N slots | 38.2% | 38.0% | 11.9% | 2.79 | 4.25 | -4.1% | 28 | 9.32 | 1.36 | 93.8% | 90.9% | 725 | 4,500 | 45 |
| WeightedTrend(21/63/126/252) | 40.4% | 40.2% | 16.1% | 2.18 | 3.39 | -5.9% | 35 | 6.81 | 2.13 | 99.2% | 77.9% | 175 | 7,379 | 74 |

```json
{
 "period": "2021 (bull)",
 "costs": {
  "commission_bps": 10.0,
  "slippage_bps": 5.0
 },
 "agent": {
  "provider": "gemini",
  "model": "gemini-3.5-flash-lite",
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
 "prompt_tokens": 175641,
 "completion_tokens": 9107,
 "retries": 0,
 "fallbacks": 0,
 "repairs": 38,
 "mean_exposure": 0.9096153846153847
}
```
