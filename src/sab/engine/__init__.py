"""Backtest engine (M2).

Event-driven daily loop: bar t close -> strategy emits orders -> fills at bar t+1 OPEN
(never same-bar) -> portfolio accounting. See `backtest.py` for the timeline and
`docs/decisions.md` for the rules (integer shares, no shorting, costs).
"""

from sab.engine.backtest import BacktestConfig, BacktestResult, Context, Strategy, run_backtest
from sab.engine.costs import CostModel, DefaultCost, ZeroCost
from sab.engine.market import MarketData, load_market
from sab.engine.types import Fill, Order, PortfolioState, Position

__all__ = [
    "BacktestConfig", "BacktestResult", "Context", "Strategy", "run_backtest",
    "CostModel", "DefaultCost", "ZeroCost", "MarketData", "load_market",
    "Fill", "Order", "PortfolioState", "Position",
]
