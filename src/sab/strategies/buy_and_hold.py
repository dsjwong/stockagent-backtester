"""Buy-and-hold benchmarks.

* EqualWeight(rebalance=True)  -- target 1/N in each name on every rebalance bar (the
  engine's rebalance frequency, monthly by default). Official benchmark ("EW-15").
* EqualWeight(rebalance=False) -- 1/N once on the first decision bar, then never trades
  (weights drift with prices).
* SpyReference                 -- 100% SPY, bought once. Reference line only; SPY is not in
  the tradeable universe.
"""

from __future__ import annotations

from sab.data.universe import BENCHMARK, TICKERS
from sab.engine.backtest import Context
from sab.engine.types import Order


class EqualWeight:
    def __init__(self, tickers: tuple[str, ...] | list[str] = TICKERS, rebalance: bool = True) -> None:
        self.tickers = list(tickers)
        self.rebalance = rebalance
        self.name = "EW-hold" if not rebalance else "EW-rebal"
        self._invested = False

    def on_bar(self, ctx: Context) -> list[Order]:
        if self._invested and not self.rebalance:
            return []
        self._invested = True
        w = 1.0 / len(self.tickers)
        return [Order.target(t, w) for t in self.tickers]


class SpyReference:
    name = "SPY"

    def __init__(self) -> None:
        self._invested = False

    def on_bar(self, ctx: Context) -> list[Order]:
        if self._invested:
            return []
        self._invested = True
        return [Order.target(BENCHMARK, 1.0)]
