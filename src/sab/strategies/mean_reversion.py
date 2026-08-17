"""Mean reversion: buy the names that are stretched below their own moving average.

Signal (per ticker, computed on adjusted closes up to the current bar):
    z_i = (P_i - SMA_N(P_i)) / STD_N(P_i)              N = 20 (one trading month)
Rule: go long every ticker with z_i < z_entry (default -1, i.e. one rolling sigma below
its 20-day mean -- the classic Bollinger-band entry), EQUAL-WEIGHT among the selected
names, everything else in cash. If nothing qualifies the portfolio is 100% cash.

Why the threshold variant rather than "always hold the bottom-k": bottom-k is always
fully invested and its results are dominated by universe beta rather than by the reversion
signal; the threshold rule is the honest "trade only when there is a dip" strategy and its
exposure (usually well below 100%) is part of the finding. `max_names` optionally caps the
number of names (most negative z first); default None = all qualifying.

Default rebalance: weekly (first bar of each ISO week). Warmup = N bars.
Parameters are textbook defaults, NOT tuned on the reporting periods (decisions.md).
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd

from sab.engine.backtest import Context
from sab.engine.types import Order
from sab.strategies._base import equal_weight, target_orders, universe


class MeanReversion:
    rebalance = "weekly"

    def __init__(self, tickers: Iterable[str] | None = None, lookback: int = 20,
                 z_entry: float = -1.0, max_names: int | None = None) -> None:
        if lookback < 2:
            raise ValueError("lookback must be >= 2")
        self.tickers = universe(tickers)
        self.lookback = int(lookback)
        self.z_entry = float(z_entry)
        self.max_names = max_names
        self.name = f"MeanRev(N={lookback},z<{z_entry:g})"
        self.warmup = self.lookback

    @property
    def params(self) -> dict:
        return {"lookback": self.lookback, "z_entry": self.z_entry, "max_names": self.max_names,
                "selection": "all with z < z_entry, equal-weight; cash otherwise", "rebalance": self.rebalance}

    def zscores(self, ctx: Context) -> pd.Series:
        px = ctx.prices(self.lookback)[self.tickers]
        if len(px) < self.lookback:
            return pd.Series(dtype=float)
        sma = px.mean()
        sd = px.std(ddof=1)
        z = (px.iloc[-1] - sma) / sd
        return z.replace([float("inf"), float("-inf")], float("nan")).dropna()

    def on_bar(self, ctx: Context) -> list[Order]:
        z = self.zscores(ctx)
        selected = z[z < self.z_entry].sort_values()
        if self.max_names is not None:
            selected = selected.iloc[: self.max_names]
        return target_orders(ctx, equal_weight(selected.index), self.tickers)
