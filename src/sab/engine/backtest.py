"""Event-driven daily backtest loop.

Timeline for bar index t (0-based over the MarketData dates):

    close[t]  : portfolio marked (equity_t = cash + sum qty * close_t)         -> equity curve
                strategy sees ONLY bars <= t via Context, emits orders (rebalance bars only)
    open[t+1] : broker fills those orders with costs, using equity_t for sizing
    close[t+1]: next mark ... and so on

So an order can never fill on the bar it was decided on. The last bar never emits orders
(there is no next open). Bars before `warmup` are marked but no strategy call is made.

Lookahead is prevented BY CONSTRUCTION: Context is built from `iloc[:t+1]` slices of the
open/close frames, so the strategy has no object through which future rows are reachable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

import pandas as pd

from sab.engine.broker import Broker
from sab.engine.costs import CostModel, DefaultCost
from sab.engine.market import MarketData
from sab.engine.types import Fill, Order, PortfolioState

Rebalance = Literal["daily", "weekly", "monthly"]


class Context:
    """Point-in-time view handed to a strategy: history up to and including the current bar."""

    __slots__ = ("_close", "_open", "portfolio", "bar_index", "cost_model")

    def __init__(
        self,
        close_hist: pd.DataFrame,
        open_hist: pd.DataFrame,
        portfolio: PortfolioState,
        bar_index: int,
        cost_model: CostModel,
    ) -> None:
        self._close = close_hist          # rows [0 .. t], nothing after
        self._open = open_hist
        self.portfolio = portfolio
        self.bar_index = bar_index
        self.cost_model = cost_model

    @property
    def date(self) -> pd.Timestamp:
        return self._close.index[-1]

    @property
    def tickers(self) -> list[str]:
        return list(self._close.columns)

    def prices(self, lookback: int | None = None) -> pd.DataFrame:
        """Adjusted closes ending at the current bar. `lookback` = number of rows (max)."""
        return self._close if lookback is None else self._close.iloc[-lookback:]

    def opens(self, lookback: int | None = None) -> pd.DataFrame:
        return self._open if lookback is None else self._open.iloc[-lookback:]

    def close(self) -> pd.Series:
        """Current bar's closes (the prices the portfolio is marked at)."""
        return self._close.iloc[-1]

    def weights(self) -> dict[str, float]:
        return self.portfolio.weights(self.close())

    def __len__(self) -> int:
        return len(self._close)


class Strategy(Protocol):
    name: str

    def on_bar(self, ctx: Context) -> list[Order]: ...


@dataclass(frozen=True)
class BacktestConfig:
    initial_cash: float = 1_000_000.0
    rebalance: Rebalance = "monthly"
    warmup: int = 0
    cost_model: CostModel = DefaultCost

    def to_dict(self) -> dict[str, Any]:
        return {
            "initial_cash": self.initial_cash,
            "rebalance": self.rebalance,
            "warmup": self.warmup,
            **self.cost_model.describe(),
        }


@dataclass
class BacktestResult:
    name: str
    equity: pd.Series                 # daily, marked at close
    cash: pd.Series
    weights: pd.DataFrame             # daily position weights (qty * close / equity)
    turnover: pd.Series               # per fill date: sum |notional| / equity_at_decision / 2
    trades: list[Fill]
    config: dict[str, Any]
    broker_events: list[str] = field(default_factory=list)
    decisions: pd.DataFrame | None = None   # per-decision log for strategies that keep one (LLM agents)

    @property
    def exposure(self) -> pd.Series:
        return self.weights.sum(axis=1)

    def trades_frame(self) -> pd.DataFrame:
        cols = ["date", "ticker", "side", "qty", "price", "commission", "slippage"]
        if not self.trades:
            return pd.DataFrame(columns=cols)
        return pd.DataFrame([{c: getattr(f, c) for c in cols} for f in self.trades])

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "config": self.config,
            "equity": {str(d.date()): float(v) for d, v in self.equity.items()},
            "n_trades": len(self.trades),
            "broker_events": self.broker_events,
        }

    def save(self, directory: str | Path, name: str | None = None) -> Path:
        """Write equity/cash/turnover, weights, trades (parquet) + config.json under
        directory/<name or self.name>. `name` lets callers pass a filesystem-safe slug."""
        d = Path(directory) / (name or self.name)
        d.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"equity": self.equity, "cash": self.cash, "turnover": self.turnover}).to_parquet(d / "equity.parquet")
        self.weights.to_parquet(d / "weights.parquet")
        self.trades_frame().to_parquet(d / "trades.parquet")
        (d / "config.json").write_text(json.dumps({"name": self.name, **self.config}, indent=2))
        if self.decisions is not None:
            dec = self.decisions.copy()
            for c in dec.columns:                       # dict/None cells -> JSON strings for parquet
                if dec[c].map(lambda v: isinstance(v, (dict, list))).any():
                    dec[c] = dec[c].map(lambda v: json.dumps(v) if isinstance(v, (dict, list)) else v)
            dec.to_parquet(d / "decisions.parquet")
        return d


def rebalance_bars(dates: pd.DatetimeIndex, freq: Rebalance, warmup: int) -> set[int]:
    """Bar indices on which the strategy is asked for orders.

    Always includes `warmup` (first decision bar). Then: every bar (daily), the first bar of
    each new ISO week (weekly) or of each new calendar month (monthly). The last bar is
    excluded everywhere (no next open to fill at).
    """
    last = len(dates) - 1
    if freq == "daily":
        return {t for t in range(warmup, last)}
    key = (lambda d: (d.isocalendar().year, d.isocalendar().week)) if freq == "weekly" else (lambda d: (d.year, d.month))
    bars = set()
    for t in range(warmup, last):
        if t == warmup or key(dates[t]) != key(dates[t - 1]):
            bars.add(t)
    return bars


def run_backtest(
    strategy: Strategy,
    market: MarketData,
    config: BacktestConfig = BacktestConfig(),
    name: str | None = None,
) -> BacktestResult:
    dates = market.dates
    T = len(dates)
    if config.warmup < 0 or config.warmup >= T - 1:
        raise ValueError(f"warmup={config.warmup} leaves no tradeable bars (T={T})")

    broker = Broker(config.cost_model)
    state = PortfolioState(cash=config.initial_cash)
    decision_bars = rebalance_bars(dates, config.rebalance, config.warmup)

    equity = pd.Series(index=dates, dtype=float)
    cash = pd.Series(index=dates, dtype=float)
    weights = pd.DataFrame(0.0, index=dates, columns=market.tickers)
    turnover = pd.Series(0.0, index=dates)
    trades: list[Fill] = []

    pending: list[Order] = []
    equity_at_decision = config.initial_cash
    for t in range(T):
        # 1) fills at this bar's OPEN for orders decided at the previous close
        if pending:
            fills = broker.execute(pending, state, market.open.iloc[t], dates[t])
            trades.extend(fills)
            if fills and equity_at_decision > 0:
                turnover.iloc[t] = sum(f.notional for f in fills) / equity_at_decision / 2
            pending = []

        # 2) mark at this bar's CLOSE
        close_t = market.close.iloc[t]
        state.mark(dates[t], close_t)
        equity.iloc[t] = state.equity
        cash.iloc[t] = state.cash
        for tk, w in state.weights(close_t).items():
            weights.iloc[t, weights.columns.get_loc(tk)] = w

        # 3) strategy decides on rebalance bars, seeing only bars <= t
        if t in decision_bars:
            ctx = Context(
                market.close.iloc[: t + 1], market.open.iloc[: t + 1], state, t, config.cost_model
            )
            pending = list(strategy.on_bar(ctx))
            equity_at_decision = state.equity

    return BacktestResult(
        name=name or getattr(strategy, "name", type(strategy).__name__),
        equity=equity,
        cash=cash,
        weights=weights,
        turnover=turnover,
        trades=trades,
        config={**config.to_dict(), "strategy": getattr(strategy, "name", type(strategy).__name__),
                "params": dict(getattr(strategy, "params", {}) or {}),
                "start": str(dates[0].date()), "end": str(dates[-1].date()), "n_bars": T,
                "n_scaled_down": broker.n_scaled_down, "n_sell_capped": broker.n_sell_capped},
        broker_events=broker.events,
        decisions=strategy.decisions_frame() if hasattr(strategy, "decisions_frame") else None,
    )
