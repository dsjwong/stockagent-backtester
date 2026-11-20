# Forward test (live paper trading)

## Hypothesis (stated before any forward decision existed)

The historical study found that the same model, given real tickers and dates, behaved very differently from
the same model given anonymised prices (2022: +11.9% real vs +1.6% anon; 8/8 vs 0/8 all-cash draws on
2022-06-13), and read that gap as **leakage**: the model remembers the period. After the model's training
cutoff there is nothing to remember. So:

* **H1 (leakage)**: if the historical gap was leakage, the forward real-minus-anon gap should **collapse
  toward zero** (and its sign/size should stop being systematic).
* **H0 (not only leakage)**: if the gap persists forward, the real prompt changes behaviour for reasons
  other than memory of the period (e.g. general sector priors or prompt-token effects), and the leakage
  reading of the historical gap is weakened.

Nothing is tuned: model id, prompt format, weekly cadence, seed and costs are pinned in `forward/config.json`
(same as the historical study). Each decision is appended to `forward/decisions.jsonl` (hash-chained) and committed
before the next US open; `git log --format='%H %cI' -- forward/decisions.jsonl` gives the commit timestamps.
The model's training cutoff is the vendor's to state; this repo does not assume one.

Start: **2026-11-05**; model `gemini-3.5-flash-lite` (provider gemini), cadence weekly, T=0.0, seed 0, costs 10+5 bps/side, rf=0. Hash chain verified over 6 records.

Bars: 2026-11-05 .. 2026-11-19 (11); decisions: 3 anon / 3 real; model id actually served: gemini-3.5-flash-lite.

## Equity since start

| strategy (rf=0) | CAGR | TotalRet | Vol | Sharpe | Sortino | MaxDD | DD_days | Calmar | Turnover | Exposure | WinRate | Trades | Costs | Costs_bps |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LLM anon | 28.8% | 1.0% | 14.1% | 1.85 | 2.94 | -2.8% | 3 | 10.41 | 11.68 | 90.7% | 100.0% | 14 | 1,530 | 15 |
| LLM real | -25.7% | -1.2% | 17.5% | -1.62 | -2.10 | -4.5% | 4 | -5.72 | 16.62 | 83.1% | 77.8% | 21 | 2,204 | 22 |
| EW-15 (monthly rebal) | -62.1% | -3.8% | 16.6% | -5.78 | -6.14 | -5.9% | 4 | -10.48 | 11.42 | 90.7% |  | 15 | 1,495 | 15 |
| SPY | -50.1% | -2.7% | 20.9% | -3.23 | -4.48 | -5.8% | 6 | -8.67 | 11.44 | 90.9% |  | 1 | 1,497 | 15 |

![forward equity](fig/forward.png)

## Anon vs real gap

Mean daily return difference (real - anon), annualised: **-54.4%** over 10 days.
95% CI withheld until n >= 60 trading days (n = 10); any reading now is noise.

For scale, the historical 2022 gap was +10.3 pp over one year (real +11.9% vs anon +1.6%), same models/prompt.

## Decisions (exposure, number of names held)

| date | anon exp | anon names | real exp | real names | decided at (UTC) |
|---|---|---|---|---|---|
| 2026-11-05 | 1.00 | 5 | 0.90 | 6 | 2026-11-06T02:38:24+00:00 |
| 2026-11-09 | 1.00 | 5 | 0.90 | 6 | 2026-11-10T04:12:54+00:00 |
| 2026-11-16 | 1.00 | 5 | 0.95 | 10 | 2026-11-17T04:40:28+00:00 |

