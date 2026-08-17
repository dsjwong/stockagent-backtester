"""Cross-sectional momentum: top-k by 12-1 month return, equal-weight, monthly.

Signal: mom_i = P_i[t - 21] / P_i[t - 252] - 1   (12-month return skipping the most recent
month, the Jegadeesh-Titman / Fama-French "12-1" convention that avoids the 1-month
reversal). Rule: hold the top `k` (default 5 of 15) equal-weight; always fully invested
(it is a *relative* strategy, so it stays long in bear markets by design -- compare with
the absolute-momentum WeightedTrend). Optional `long_only_positive=True` would drop
names with negative momentum; off by default to keep the textbook form.
Default rebalance: monthly. Warmup = 252 bars.
Parameters are textbook defaults, NOT tuned on the reporting periods (decisions.md).
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd

from sab.engine.backtest import Context
from sab.engine.types import Order
from sab.strategies._base import equal_weight, target_orders, universe


class MomentumTopK:
    rebalance = "monthly"

    def __init__(self, tickers: Iterable[str] | None = None, k: int = 5,
                 lookback: int = 252, skip: int = 21, long_only_positive: bool = False) -> None:
        if not 0 <= skip < lookback:
            raise ValueError("need 0 <= skip < lookback")
        self.tickers = universe(tickers)
        self.k, self.lookback, self.skip = int(k), int(lookback), int(skip)
        self.long_only_positive = long_only_positive
        self.name = f"MomTop{k}(12-1m)"
        self.warmup = self.lookback

    @property
    def params(self) -> dict:
        return {"k": self.k, "lookback": self.lookback, "skip": self.skip,
                "long_only_positive": self.long_only_positive, "rebalance": self.rebalance}

    def momentum(self, ctx: Context) -> pd.Series:
        n = self.lookback + 1
        px = ctx.prices(n)[self.tickers]
        if len(px) < n:
            return pd.Series(dtype=float)
        return px.iloc[-1 - self.skip] / px.iloc[0] - 1

    def on_bar(self, ctx: Context) -> list[Order]:
        mom = self.momentum(ctx)
        if mom.empty:
            return []
        if self.long_only_positive:
            mom = mom[mom > 0]
        top = mom.sort_values(ascending=False).iloc[: self.k]
        return target_orders(ctx, equal_weight(top.index), self.tickers)
