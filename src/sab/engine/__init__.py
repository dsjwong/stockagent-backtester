"""Backtest engine (M2 — not yet implemented).

Will contain: the event-driven daily loop (bar t -> strategy emits orders -> fill at
bar t+1 OPEN, never same-bar), Portfolio accounting (cash, positions, equity curve),
and the cost model (commission 10 bps + slippage 5 bps per side; see docs/decisions.md).
Strategies must only ever receive data up to and including the decision bar.
"""
