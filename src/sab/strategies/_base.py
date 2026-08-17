"""Shared helpers for the classical strategies (M3).

Every classical strategy:
* trades only `self.tickers` (default: the 15-name universe; SPY is never traded even
  when it is present in the market frame),
* reads history ONLY through `ctx.prices(lookback)` (adjusted closes up to and including
  the current bar) -- nothing else on the Context is touched, so lookahead safety is
  inherited from the engine's `iloc[:t+1]` slicing,
* exposes `params` (surfaced in the result config), `warmup` (bars of history it needs
  before its first decision; the runner passes it to `BacktestConfig`) and `rebalance`
  (the textbook default frequency for that strategy),
* emits target-weight orders in [0, 1] whose sum is <= 1, and an explicit weight-0 order
  for every held name it wants to exit (so positions are actually flattened).
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd

from sab.data.universe import TICKERS
from sab.engine.backtest import Context
from sab.engine.types import Order


def universe(tickers: Iterable[str] | None) -> list[str]:
    return list(tickers) if tickers is not None else list(TICKERS)


def target_orders(ctx: Context, weights: pd.Series | dict[str, float], tickers: list[str]) -> list[Order]:
    """Turn a {ticker: weight} mapping into orders. Names not in `weights` (or at 0) that are
    currently held get an explicit 0 target so they are sold; tiny weights are dropped.
    Weights are clipped to [0, 1] and, if they sum to > 1 (float noise), rescaled."""
    w = pd.Series(weights, dtype=float).reindex(tickers).fillna(0.0).clip(lower=0.0)
    total = float(w.sum())
    if total > 1.0:
        w = w / total
    held = {t for t, q in ((t, ctx.portfolio.qty(t)) for t in tickers) if q > 0}
    orders: list[Order] = []
    for t in tickers:
        wt = float(w[t])
        if wt >= 1e-6:
            orders.append(Order.target(t, min(wt, 1.0)))
        elif t in held:
            orders.append(Order.target(t, 0.0))
    return orders


def equal_weight(selected: Iterable[str]) -> dict[str, float]:
    sel = list(selected)
    return {t: 1.0 / len(sel) for t in sel} if sel else {}
