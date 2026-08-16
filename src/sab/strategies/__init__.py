"""Strategies. Interface: `on_bar(ctx: Context) -> list[Order]` (see sab.engine.backtest).

M2: buy-and-hold benchmarks only. M3: mean reversion, trend following. M4: LLM agents
with a (prompt -> decision) cache.
"""
