# LLM(gemini/gemini-3.5-flash-lite, anon) on full (2019-01 .. 2026-08)

costs 10+5 bps/side, rf=0

| strategy (rf=0) | CAGR | TotalRet | Vol | Sharpe | Sortino | MaxDD | DD_days | Calmar | Turnover | Exposure | WinRate | Trades | Costs | Costs_bps |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LLM(gemini/gemini-3.5-flash-lite, anon) | 20.4% | 310.9% | 19.9% | 1.04 | 1.49 | -25.0% | 393 | 0.82 | 9.36 | 95.8% | 70.1% | 2635 | 605,495 | 6055 |
| EW-rebal (monthly) | 22.7% | 374.2% | 18.1% | 1.23 | 1.76 | -31.2% | 294 | 0.73 | 0.36 | 99.6% | 93.8% | 1373 | 18,088 | 181 |
| EW-hold | 31.4% | 694.7% | 24.9% | 1.22 | 1.79 | -30.7% | 354 | 1.02 | 0.07 | 99.9% |  | 15 | 1,497 | 15 |
| SPY (reference) | 17.9% | 249.1% | 19.4% | 0.95 | 1.34 | -33.7% | 488 | 0.53 | 0.07 | 99.9% |  | 1 | 1,498 | 15 |
| Trend(SMA50>SMA200) | 21.6% | 340.3% | 17.9% | 1.18 | 1.70 | -28.3% | 291 | 0.76 | 2.57 | 99.6% | 87.3% | 4250 | 135,699 | 1357 |
| Trend(SMA50>SMA200), 1/N slots | 14.6% | 182.2% | 12.8% | 1.13 | 1.61 | -23.7% | 302 | 0.62 | 1.18 | 70.4% | 89.1% | 4220 | 48,009 | 480 |
| WeightedTrend(21/63/126/252) | 26.8% | 508.0% | 21.6% | 1.21 | 1.76 | -28.6% | 192 | 0.94 | 2.45 | 98.2% | 70.6% | 1121 | 168,788 | 1688 |

```json
{
 "period": "full (2019-01 .. 2026-08)",
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
 "n_decisions": 398,
 "n_model_calls": 399,
 "n_provider_calls_this_run": 146,
 "cache_hits": 253,
 "prompt_tokens": 1374458,
 "completion_tokens": 63555,
 "retries": 1,
 "fallbacks": 0,
 "repairs": 2116,
 "mean_exposure": 0.9611809045226131
}
```
