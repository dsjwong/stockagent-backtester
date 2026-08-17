"""Trend following: the classic dual moving-average filter (golden cross / death cross).

Signal (per ticker): uptrend  <=>  SMA_fast(P) > SMA_slow(P), fast = 50, slow = 200 bars.
Rule (sizing="equal", default): EQUAL-WEIGHT long every ticker in an uptrend; names not in
an uptrend get 0; if no name is in an uptrend the portfolio is 100% cash. Note this is fully
invested whenever at least one name trends up (concentrated in the trenders).
Variant (sizing="slots"): each ticker owns a fixed 1/N slot that is either long (uptrend)
or cash -- the classic per-name trend *filter*, whose exposure falls in broad drawdowns.
Both are reported in M3; there is no shorting (engine rule).

Chosen over the 12-month time-series-momentum variant because it is the most widely
quoted textbook rule and needs no extra parameter (skip-month, threshold). Default
rebalance: weekly. Warmup = slow bars (200 ~ 10 months), so on the full 2018-2026 run
the first decision is in mid-October 2018; the 2021 / 2022 sub-period runs are therefore
started with 200 bars of *pre-period* history (the runner handles that), not from cash
with no signal.
Parameters are textbook defaults, NOT tuned on the reporting periods (decisions.md).
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd

from sab.engine.backtest import Context
from sab.engine.types import Order
from sab.strategies._base import equal_weight, target_orders, universe


class TrendFollowing:
    rebalance = "weekly"

    def __init__(self, tickers: Iterable[str] | None = None, fast: int = 50, slow: int = 200,
                 sizing: str = "equal") -> None:
        if not 1 <= fast < slow:
            raise ValueError("need 1 <= fast < slow")
        if sizing not in ("equal", "slots"):
            raise ValueError("sizing must be 'equal' or 'slots'")
        self.tickers = universe(tickers)
        self.fast, self.slow = int(fast), int(slow)
        self.sizing = sizing
        self.name = f"Trend(SMA{fast}>SMA{slow})" + (", 1/N slots" if sizing == "slots" else "")
        self.warmup = self.slow

    @property
    def params(self) -> dict:
        return {"fast": self.fast, "slow": self.slow, "sizing": self.sizing,
                "selection": ("equal-weight all with SMA_fast > SMA_slow; cash otherwise" if self.sizing == "equal"
                              else "1/N in each name with SMA_fast > SMA_slow, its slot in cash otherwise"),
                "rebalance": self.rebalance}

    def in_uptrend(self, ctx: Context) -> pd.Series:
        px = ctx.prices(self.slow)[self.tickers]
        if len(px) < self.slow:
            return pd.Series(False, index=self.tickers)
        return px.iloc[-self.fast:].mean() > px.mean()

    def on_bar(self, ctx: Context) -> list[Order]:
        up = self.in_uptrend(ctx)
        if self.sizing == "slots":
            w = {t: 1.0 / len(self.tickers) for t in up[up].index}
        else:
            w = equal_weight(up[up].index)
        return target_orders(ctx, w, self.tickers)
