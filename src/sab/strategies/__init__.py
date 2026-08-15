"""Strategies (M3/M4 — not yet implemented).

Common interface: given a point-in-time view of prices (up to the decision bar) and the
current portfolio, return target weights / orders. Planned: buy-and-hold equal-weight
(benchmark), mean reversion, trend following, weighted trend (Python re-implementations
of the C++ project), and LLM agents ported from StockAgent with a (prompt -> decision) cache.
"""
