"""Weighted Trend: size positions in proportion to positive momentum strength.

This is the "Weighted Trend" classical baseline from StockAgent, rewritten in Python:
    score_i = mean over horizons h in {21, 63, 126, 252} bars of (P_i / P_i[-h] - 1)
              (1 / 3 / 6 / 12-month simple returns, equally blended)
    raw_i   = max(score_i, 0)
    w_i     = raw_i / sum_j raw_j          (if sum > 0, i.e. fully invested; else 100% cash)
Optionally `max_weight` caps any single name (default 0.25) with the excess redistributed
pro rata over the uncapped names (iterated), so one runaway winner cannot become the whole
book. Names with a non-positive blended score get 0.

Default rebalance: monthly (weights move slowly; weekly would only add turnover).
Warmup = 252 bars (longest horizon).
Parameters are textbook defaults, NOT tuned on the reporting periods (decisions.md).
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd

from sab.engine.backtest import Context
from sab.engine.types import Order
from sab.strategies._base import target_orders, universe


def cap_weights(w: pd.Series, cap: float, iters: int = 20) -> pd.Series:
    """Cap each weight at `cap`, redistribute the excess pro rata over uncapped names.
    If cap * n < 1 the total simply ends up below 1 (rest is cash)."""
    w = w.copy()
    for _ in range(iters):
        over = w > cap + 1e-12
        if not over.any():
            break
        excess = float((w[over] - cap).sum())
        w[over] = cap
        free = (~over) & (w > 0)
        if not free.any():
            break
        w[free] += excess * w[free] / float(w[free].sum())
    return w.clip(upper=cap)


class WeightedTrend:
    rebalance = "monthly"

    def __init__(self, tickers: Iterable[str] | None = None,
                 horizons: tuple[int, ...] = (21, 63, 126, 252), max_weight: float | None = 0.25) -> None:
        if not horizons or min(horizons) < 1:
            raise ValueError("horizons must be positive ints")
        self.tickers = universe(tickers)
        self.horizons = tuple(int(h) for h in horizons)
        self.max_weight = max_weight
        self.name = f"WeightedTrend({'/'.join(str(h) for h in self.horizons)})"
        self.warmup = max(self.horizons)

    @property
    def params(self) -> dict:
        return {"horizons": list(self.horizons), "max_weight": self.max_weight,
                "weighting": "blend of horizon returns, clipped at 0, normalised to 1 (cash if none > 0)",
                "rebalance": self.rebalance}

    def scores(self, ctx: Context) -> pd.Series:
        n = max(self.horizons) + 1
        px = ctx.prices(n)[self.tickers]
        if len(px) < n:
            return pd.Series(dtype=float)
        last = px.iloc[-1]
        rets = [last / px.iloc[-1 - h] - 1 for h in self.horizons]
        return pd.concat(rets, axis=1).mean(axis=1)

    def target_weights(self, ctx: Context) -> pd.Series:
        s = self.scores(ctx)
        if s.empty:
            return s
        raw = s.clip(lower=0.0)
        tot = float(raw.sum())
        if tot <= 0:
            return raw * 0.0
        w = raw / tot
        if self.max_weight is not None:
            w = cap_weights(w, self.max_weight)
        return w

    def on_bar(self, ctx: Context) -> list[Order]:
        return target_orders(ctx, self.target_weights(ctx), self.tickers)
