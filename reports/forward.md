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

Bars: 2026-11-05 .. 2026-11-17 (9); decisions: 3 anon / 3 real; model id actually served: gemini-3.5-flash-lite.

## Equity since start

| strategy (rf=0) | CAGR | TotalRet | Vol | Sharpe | Sortino | MaxDD | DD_days | Calmar | Turnover | Exposure | WinRate | Trades | Costs | Costs_bps |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LLM anon | 162.0% | 3.1% | 11.4% | 8.55 | 20.86 | -0.7% | 1 | 217.84 | 14.27 | 88.7% | 100.0% | 14 | 1,530 | 15 |
| LLM real | 110.4% | 2.4% | 12.2% | 6.15 | 13.41 | -1.0% | 2 | 105.56 | 20.31 | 80.4% | 77.8% | 21 | 2,204 | 22 |
| EW-15 (monthly rebal) | -6.6% | -0.2% | 13.0% | -0.47 | -0.62 | -2.4% | 2 | -2.70 | 13.96 | 88.7% |  | 15 | 1,495 | 15 |
| SPY | -13.9% | -0.5% | 21.1% | -0.61 | -1.03 | -3.6% | 4 | -3.85 | 13.98 | 88.8% |  | 1 | 1,497 | 15 |

![forward equity](fig/forward.png)

## Anon vs real gap

Mean daily return difference (real - anon), annualised: **-21.9%** over 8 days.
95% CI withheld until n >= 60 trading days (n = 8); any reading now is noise.

For scale, the historical 2022 gap was +10.3 pp over one year (real +11.9% vs anon +1.6%), same models/prompt.

## Decisions (exposure, number of names held)

| date | anon exp | anon names | real exp | real names | decided at (UTC) |
|---|---|---|---|---|---|
| 2026-11-05 | 1.00 | 5 | 0.90 | 6 | 2026-11-06T02:38:24+00:00 |
| 2026-11-09 | 1.00 | 5 | 0.90 | 6 | 2026-11-10T04:12:54+00:00 |
| 2026-11-16 | 1.00 | 5 | 0.95 | 10 | 2026-11-17T04:40:28+00:00 |

