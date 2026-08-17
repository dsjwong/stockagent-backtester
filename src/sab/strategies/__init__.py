"""Strategies. Interface: `on_bar(ctx: Context) -> list[Order]` (see sab.engine.backtest).

M2: buy-and-hold benchmarks. M3: classical strategies (mean reversion, dual-SMA trend
following, weighted trend, 12-1 momentum top-k) -- each exposes `params`, `warmup`,
`rebalance`. M4: LLM agents with a (prompt -> decision) cache.
"""

from sab.strategies.buy_and_hold import EqualWeight, SpyReference
from sab.strategies.mean_reversion import MeanReversion
from sab.strategies.momentum_top_k import MomentumTopK
from sab.strategies.trend_following import TrendFollowing
from sab.strategies.weighted_trend import WeightedTrend

CLASSICAL = (MeanReversion, TrendFollowing, WeightedTrend, MomentumTopK)

__all__ = ["EqualWeight", "SpyReference", "MeanReversion", "TrendFollowing", "WeightedTrend", "MomentumTopK", "CLASSICAL"]
