# LLM(gemini/gemini-3.5-flash-lite, anon) on 2021 (bull)

costs 10+5 bps/side, rf=0

| strategy (rf=0) | CAGR | TotalRet | Vol | Sharpe | Sortino | MaxDD | DD_days | Calmar | Turnover | Exposure | WinRate | Trades | Costs | Costs_bps |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LLM(gemini/gemini-3.5-flash-lite, anon) | 41.5% | 41.3% | 15.7% | 2.29 | 3.60 | -7.4% | 48 | 5.58 | 9.25 | 94.4% | 81.5% | 386 | 34,138 | 341 |
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
  "mode": "anon",
  "lookback": 60,
  "temperature": 0.0,
  "seed": 0,
  "max_tokens": 400,
  "rebalance": "weekly",
  "warmup": 127,
  "max_retries": 1,
  "anon_mapping": {
   "seed": 0,
   "ticker_to_anon": {
    "NVDA": "Stock 01",
    "UNH": "Stock 02",
    "AMZN": "Stock 03",
    "PFE": "Stock 04",
    "AAPL": "Stock 05",
    "GOOGL": "Stock 06",
    "XOM": "Stock 07",
    "JPM": "Stock 08",
    "WMT": "Stock 09",
    "PG": "Stock 10",
    "BAC": "Stock 11",
    "JNJ": "Stock 12",
    "KO": "Stock 13",
    "CVX": "Stock 14",
    "MSFT": "Stock 15"
   }
  }
 },
 "n_decisions": 52,
 "n_model_calls": 52,
 "n_provider_calls_this_run": 52,
 "cache_hits": 0,
 "prompt_tokens": 180158,
 "completion_tokens": 8192,
 "retries": 0,
 "fallbacks": 0,
 "repairs": 292,
 "mean_exposure": 0.9498076923076925
}
```
