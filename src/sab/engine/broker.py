"""Broker: turns orders decided at bar t's close into fills at bar t+1's open.

Rules (decisions.md #5, #6, #10):
* Target-weight orders -> integer share qty = floor(w * equity_t / buy_cost_per_share(open_{t+1})),
  where equity_t is the portfolio marked at the DECISION-time close and open_{t+1} is the
  next bar's open. Budgeting with the all-in buy price (open + slippage + commission) means a
  fully-invested set of weights normally fits in cash without scaling.
* Integer shares only; the cash residual sits idle (no fractional shares, no interest).
* No shorting: sells are capped at the shares held; target weights are validated >= 0
  and their sum must be <= 1 (+1e-9).
* Sells are executed before buys (they free cash). If the buys would still push cash
  negative, ALL buy quantities are scaled down proportionally (floored) and the event is
  logged and counted -- the portfolio never borrows.
* Every order is a market order and always fully fills at the open (large-caps, small
  notional relative to volume; documented limitation).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import pandas as pd

from sab.engine.costs import CostModel
from sab.engine.types import Fill, Order, PortfolioState, Side

log = logging.getLogger(__name__)


@dataclass
class Broker:
    cost_model: CostModel
    n_scaled_down: int = 0
    n_sell_capped: int = 0
    _log: list[str] = field(default_factory=list)

    # ---- public -----------------------------------------------------------------
    def execute(
        self,
        orders: list[Order],
        state: PortfolioState,
        next_open: pd.Series,
        fill_date: pd.Timestamp,
    ) -> list[Fill]:
        """Fill `orders` (decided at `state.date`'s close) at `next_open`, mutating `state`."""
        if not orders:
            return []
        self._validate(orders, next_open)
        deltas = self._share_deltas(orders, state, next_open)
        sells = {t: -d for t, d in deltas.items() if d < 0}
        buys = {t: d for t, d in deltas.items() if d > 0}

        fills: list[Fill] = []
        for ticker, qty in sells.items():
            fills.append(self._fill(fill_date, ticker, "SELL", qty, next_open[ticker], state))
        buys = self._fit_to_cash(buys, state.cash, next_open, fill_date)
        for ticker, qty in buys.items():
            fills.append(self._fill(fill_date, ticker, "BUY", qty, next_open[ticker], state))
        return fills

    # ---- internals ---------------------------------------------------------------
    @staticmethod
    def _validate(orders: list[Order], next_open: pd.Series) -> None:
        seen: set[str] = set()
        total_w = 0.0
        for o in orders:
            if o.ticker not in next_open.index:
                raise KeyError(f"order for unknown ticker {o.ticker!r}")
            if o.ticker in seen:
                raise ValueError(f"duplicate order for {o.ticker} in the same bar")
            seen.add(o.ticker)
            if o.is_target:
                total_w += o.target_weight  # type: ignore[operator]
        if total_w > 1 + 1e-9:
            raise ValueError(f"target weights sum to {total_w:.4f} > 1 (no leverage in v1)")

    def _share_deltas(
        self, orders: list[Order], state: PortfolioState, next_open: pd.Series
    ) -> dict[str, int]:
        """Signed share change per ticker (+ buy, - sell)."""
        deltas: dict[str, int] = {}
        for o in orders:
            held = state.qty(o.ticker)
            if o.is_target:
                per_share = self.cost_model.buy_cost_per_share(float(next_open[o.ticker]))
                target_qty = math.floor(o.target_weight * state.equity / per_share)  # type: ignore[operator]
                delta = target_qty - held
            elif o.side == "BUY":
                delta = o.qty  # type: ignore[assignment]
            else:
                delta = -min(o.qty, held)  # type: ignore[type-var]
                if o.qty > held:  # type: ignore[operator]
                    self.n_sell_capped += 1
                    self._note(f"{state.date.date()} SELL {o.ticker} {o.qty} capped to {held} held")
            if delta:
                deltas[o.ticker] = int(delta)
        return deltas

    def _fit_to_cash(
        self, buys: dict[str, int], cash: float, next_open: pd.Series, fill_date: pd.Timestamp
    ) -> dict[str, int]:
        """Scale buy quantities down proportionally if their all-in cost exceeds cash."""
        if not buys:
            return buys
        cost = sum(q * self.cost_model.buy_cost_per_share(float(next_open[t])) for t, q in buys.items())
        if cost <= cash + 1e-9:
            return buys
        factor = max(cash, 0.0) / cost
        scaled = {t: math.floor(q * factor) for t, q in buys.items()}
        scaled = {t: q for t, q in scaled.items() if q > 0}
        self.n_scaled_down += 1
        self._note(f"{fill_date.date()} buys cost {cost:.2f} > cash {cash:.2f}; scaled by {factor:.4f}")
        # flooring only reduces cost, so one pass is enough
        return scaled

    def _fill(
        self,
        date: pd.Timestamp,
        ticker: str,
        side: Side,
        qty: int,
        open_price: float,
        state: PortfolioState,
    ) -> Fill:
        open_price = float(open_price)
        price = self.cost_model.fill_price(open_price, side)
        fill = Fill(
            date=date,
            ticker=ticker,
            side=side,
            qty=int(qty),
            price=price,
            commission=self.cost_model.commission(price * qty),
            slippage=abs(price - open_price) * qty,
        )
        state.apply(fill)
        return fill

    def _note(self, msg: str) -> None:
        self._log.append(msg)
        log.info(msg)

    @property
    def events(self) -> list[str]:
        return list(self._log)
