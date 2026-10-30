# Within-family check: does the 2022 real-vs-anon gap replicate on another Gemini model id?

Same engine, prompt, weekly cadence, T=0, seed 0, costs 10+5 bps, rf=0, period 2022; only the model id changes (**gemini-3.1-flash-lite** is a different release in the same flash-lite tier -- a within-family check, NOT cross-family).

| model | arm | 2022 total return | CAGR | Sharpe | MaxDD | turnover/yr | mean exposure | decisions all-cash |
|---|---|---|---|---|---|---|---|---|
| gemini-3.5-flash-lite | real | 11.9% | 12.0% | 0.70 | -15.5% | 4.5 | 93% | 0% |
| gemini-3.5-flash-lite | anon | 1.6% | 1.6% | 0.18 | -21.6% | 8.6 | 94% | 0% |
| gemini-3.1-flash-lite | real | 8.2% | 8.3% | 0.49 | -13.8% | 9.9 | 95% | 0% |
| gemini-3.1-flash-lite | anon | -0.8% | -0.8% | 0.10 | -28.7% | 13.1 | 97% | 0% |
| EW-15 rebal (benchmark) | - | -9.6% | -9.7% | -0.35 | -19.6% | 0.8 | 99% | - |

## Gap (real minus anon)

| model | return gap (pp) | Sharpe gap | weight-path correlation real/anon | decisions |
|---|---|---|---|---|
| gemini-3.5-flash-lite | +10.3 | +0.52 | 0.65 | 52/52 |
| gemini-3.1-flash-lite | +9.0 | +0.40 | 0.80 | 52/52 |

## Words in the rationales that the prompt never supplied (count of decisions, real / anon)

| term | gemini-3.5-flash-lite | gemini-3.1-flash-lite |
|---|---|---|
| energy/oil | 49 / 0 | 48 / 0 |
| tech | 21 / 0 | 28 / 0 |
| healthcare | 30 / 1 | 24 / 0 |
| defensive | 46 / 4 | 48 / 1 |
| real ticker symbol | 12 / 0 | 31 / 0 |

