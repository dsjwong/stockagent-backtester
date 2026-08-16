"""Core value types for the backtest engine: orders, fills, portfolio state.

Conventions
-----------
* Quantities are INTEGER shares. Sides are "BUY"/"SELL"; a Fill's `qty` is always positive
  and its `side` says which way.
* An Order is EITHER a target-weight order (the primary form: "hold w of equity in this
  ticker") OR a raw quantity order ("buy/sell n shares"). Never both.
* No shorting in v1: target weights are >= 0 and sells are capped at the shares held.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

Side = Literal["BUY", "SELL"]


@dataclass(frozen=True)
class Order:
    """One order emitted by a strategy at the close of bar t; fills at bar t+1's open.

    Use the constructors `Order.target(...)` or `Order.market(...)` rather than the raw
    dataclass to keep the two forms distinct.
    """

    ticker: str
    side: Side | None = None            # required for qty orders; derived for target-weight orders
    qty: int | None = None
    target_weight: float | None = None

    def __post_init__(self) -> None:
        if (self.qty is None) == (self.target_weight is None):
            raise ValueError("Order needs exactly one of qty / target_weight")
        if self.qty is not None:
            if self.side not in ("BUY", "SELL"):
                raise ValueError("qty orders need side BUY or SELL")
            if not isinstance(self.qty, int) or self.qty <= 0:
                raise ValueError(f"qty must be a positive int, got {self.qty!r}")
        if self.target_weight is not None:
            if self.target_weight < 0:
                raise ValueError(f"no shorting: target_weight must be >= 0, got {self.target_weight}")
            if self.target_weight > 1:
                raise ValueError(f"no leverage: target_weight must be <= 1, got {self.target_weight}")

    @property
    def is_target(self) -> bool:
        return self.target_weight is not None

    @classmethod
    def target(cls, ticker: str, weight: float) -> "Order":
        return cls(ticker=ticker, target_weight=float(weight))

    @classmethod
    def market(cls, ticker: str, side: Side, qty: int) -> "Order":
        return cls(ticker=ticker, side=side, qty=int(qty))


@dataclass(frozen=True)
class Fill:
    """An executed trade. `price` already includes slippage; `commission` and `slippage`
    are the dollar amounts paid (slippage = |price - open| * qty)."""

    date: pd.Timestamp
    ticker: str
    side: Side
    qty: int
    price: float
    commission: float
    slippage: float

    @property
    def notional(self) -> float:
        return self.qty * self.price

    @property
    def cash_flow(self) -> float:
        """Signed change in cash caused by this fill (negative for buys)."""
        sign = -1.0 if self.side == "BUY" else 1.0
        return sign * self.notional - self.commission

    @property
    def total_cost(self) -> float:
        return self.commission + self.slippage


@dataclass
class Position:
    ticker: str
    qty: int = 0
    avg_cost: float = 0.0        # average fill price of the open shares (incl. slippage, excl. commission)


@dataclass
class PortfolioState:
    """Cash + integer positions, marked to `equity` at `date`'s close."""

    cash: float
    positions: dict[str, Position] = field(default_factory=dict)
    equity: float = 0.0
    date: pd.Timestamp | None = None

    def qty(self, ticker: str) -> int:
        pos = self.positions.get(ticker)
        return pos.qty if pos else 0

    def holdings_value(self, close: pd.Series) -> float:
        return float(sum(p.qty * close[t] for t, p in self.positions.items() if p.qty))

    def mark(self, date: pd.Timestamp, close: pd.Series) -> None:
        """Mark equity at this bar's close (call once per bar, before decisions)."""
        self.date = date
        self.equity = self.cash + self.holdings_value(close)

    def weights(self, close: pd.Series) -> dict[str, float]:
        if self.equity <= 0:
            return {t: 0.0 for t in self.positions}
        return {t: p.qty * close[t] / self.equity for t, p in self.positions.items()}

    def apply(self, fill: Fill) -> None:
        pos = self.positions.setdefault(fill.ticker, Position(fill.ticker))
        if fill.side == "BUY":
            new_qty = pos.qty + fill.qty
            pos.avg_cost = (pos.avg_cost * pos.qty + fill.price * fill.qty) / new_qty
            pos.qty = new_qty
        else:
            if fill.qty > pos.qty:
                raise ValueError(f"cannot sell {fill.qty} {fill.ticker}: only {pos.qty} held")
            pos.qty -= fill.qty
            if pos.qty == 0:
                pos.avg_cost = 0.0
        self.cash += fill.cash_flow
        if self.cash < -1e-9:
            raise RuntimeError(f"cash went negative ({self.cash:.2f}) after {fill}")
