"""MarketData: aligned wide frames of adjusted OPEN (fills) and adjusted CLOSE (marking).

Adjusted open = Open * (Adj Close / Close) of the same bar (decisions.md #3), so fills and
marks live on the same split/dividend-adjusted scale. Built from the offline loader only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from sab.data.loader import load_ohlcv
from sab.data.universe import TICKERS


@dataclass(frozen=True)
class MarketData:
    open: pd.DataFrame     # adjusted open, index = dates, columns = tickers
    close: pd.DataFrame    # adjusted close, same shape/index/columns

    def __post_init__(self) -> None:
        if not self.open.index.equals(self.close.index):
            raise ValueError("open/close index mismatch")
        if list(self.open.columns) != list(self.close.columns):
            raise ValueError("open/close columns mismatch")
        if self.open.isna().any().any() or self.close.isna().any().any():
            raise ValueError("MarketData must not contain NaN")
        if not self.open.index.is_monotonic_increasing or self.open.index.has_duplicates:
            raise ValueError("MarketData index must be strictly increasing")

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.open.index  # type: ignore[return-value]

    @property
    def tickers(self) -> list[str]:
        return list(self.open.columns)

    def __len__(self) -> int:
        return len(self.open)

    def slice(self, start=None, end=None) -> "MarketData":
        o = self.open.loc[start:end]
        c = self.close.loc[start:end]
        return MarketData(o, c)

    def with_perturbation(self, after: pd.Timestamp, factor: float,
                          tickers: Iterable[str] | None = None) -> "MarketData":
        """Copy where every bar with date > `after` is multiplied by `factor` (open AND
        close), for `tickers` (default: all). Test helper for lookahead checks."""
        mask = self.open.index > pd.Timestamp(after)
        cols = list(tickers) if tickers is not None else self.tickers
        o, c = self.open.copy(), self.close.copy()
        o.loc[mask, cols] *= factor
        c.loc[mask, cols] *= factor
        return MarketData(o, c)


def load_market(tickers: Iterable[str] | None = None, start=None, end=None) -> MarketData:
    """Build MarketData from the committed snapshots (dates = intersection across tickers)."""
    tickers = list(tickers) if tickers is not None else list(TICKERS)
    opens, closes = {}, {}
    for t in tickers:
        df = load_ohlcv(t)
        factor = df["Adj Close"] / df["Close"]
        opens[t] = (df["Open"] * factor).rename(t)
        closes[t] = df["Adj Close"].rename(t)
    o = pd.concat(opens.values(), axis=1, join="inner").sort_index()
    c = pd.concat(closes.values(), axis=1, join="inner").sort_index()
    md = MarketData(o, c)
    if start is not None or end is not None:
        md = md.slice(pd.Timestamp(start) if start else None, pd.Timestamp(end) if end else None)
    if len(md) < 2:
        raise ValueError("need at least 2 bars")
    return md
