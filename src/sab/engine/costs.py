"""Transaction cost model: commission + slippage, both in basis points of notional.

fill price = open * (1 + slippage_bps/1e4) for buys, open * (1 - slippage_bps/1e4) for sells
commission = commission_bps/1e4 * (fill price * qty)

Defaults 10 bps commission + 5 bps slippage per side (decisions.md #6): a round trip on
$100 of stock costs 2 * (0.10 + 0.05) = $0.30, i.e. 30 bps.
"""

from __future__ import annotations

from dataclasses import dataclass

from sab.engine.types import Side


@dataclass(frozen=True)
class CostModel:
    commission_bps: float = 10.0
    slippage_bps: float = 5.0

    def fill_price(self, open_price: float, side: Side) -> float:
        slip = self.slippage_bps / 1e4
        return open_price * (1 + slip) if side == "BUY" else open_price * (1 - slip)

    def commission(self, notional: float) -> float:
        return abs(notional) * self.commission_bps / 1e4

    def buy_cost_per_share(self, open_price: float) -> float:
        """All-in cash needed to buy one share at this open (used for budgeting)."""
        p = self.fill_price(open_price, "BUY")
        return p + self.commission(p)

    def describe(self) -> dict[str, float]:
        return {"commission_bps": self.commission_bps, "slippage_bps": self.slippage_bps}


ZeroCost = CostModel(commission_bps=0.0, slippage_bps=0.0)
DefaultCost = CostModel()
