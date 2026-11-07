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

Start: **2026-11-05**; model `gemini-3.5-flash-lite` (provider gemini), cadence weekly, T=0.0, seed 0, costs 10+5 bps/side, rf=0. Hash chain verified over 2 records.

Bars: 2026-11-05 .. 2026-11-06 (2); decisions: 1 anon / 1 real; model id actually served: gemini-3.5-flash-lite.

## Equity since start

| strategy (rf=0) | CAGR | TotalRet | Vol | Sharpe | Sortino | MaxDD | DD_days | Calmar | Turnover | Exposure | WinRate | Trades | Costs | Costs_bps |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LLM anon | 3477.9% | 1.4% |  |  |  | 0.0% | 0 |  | 62.91 | 50.0% |  | 5 | 1,498 | 15 |
| LLM real | 1875.9% | 1.2% |  |  |  | 0.0% | 0 |  | 56.57 | 45.0% |  | 6 | 1,347 | 13 |
| EW-15 (monthly rebal) | 466.8% | 0.7% |  |  |  | 0.0% | 0 |  | 62.81 | 49.9% |  | 15 | 1,495 | 15 |
| SPY | -19.5% | -0.1% |  |  |  | -0.1% | 1 | -226.68 | 62.90 | 50.0% |  | 1 | 1,497 | 15 |

![forward equity](fig/forward.png)

## Anon vs real gap

Mean daily return difference (real - anon), annualised: **-60.2%** over 1 days.
95% CI withheld until n >= 60 trading days (n = 1); any reading now is noise.

For scale, the historical 2022 gap was +10.3 pp over one year (real +11.9% vs anon +1.6%), same models/prompt.

## Decisions (exposure, number of names held)

| date | anon exp | anon names | real exp | real names | decided at (UTC) |
|---|---|---|---|---|---|
| 2026-11-05 | 1.00 | 5 | 0.90 | 6 | 2026-11-06T02:38:24+00:00 |

