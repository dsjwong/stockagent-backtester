# LinkedIn post draft

Ready to paste. ~185 words, 4 hashtags. Replace `<REPO LINK>` before posting.

---

Can an LLM trade? Only if it can't cheat.

I'm a final-year CS + AI student at HKUST. I built a no-lookahead backtester and gave Gemini the same 15 US
large-caps twice: once with real tickers and dates, once anonymised — "Stock 07", "Day t", prices rebased to
100. Same engine, same prices, same seed.

Three things I found:

1. With real names, the agent "called" 2022: +11.9% while equal-weight buy-and-hold lost 9.6%. Anonymised, the
same model made +1.6%.

2. I froze one prompt (13 June 2022, all cash) and called the model 8 times per mode. Real names: 100% cash in
8/8 draws, citing a "broad market selloff". Anonymised: fully invested in 8/8. That isn't forecasting the
drawdown — it's recognising it.

3. With the leak controlled, the honest answer is a negative one — over 2019-2026 the anonymised agent returns
20.4% CAGR at Sharpe 1.04 vs 22.7% / 1.23 for equal weight, and burns 3.5 percentage points a year on turnover.

Full writeup, code and every cached prompt: <REPO LINK> — the repo goes public this week. 📉

#quant #LLM #backtesting #datascience
