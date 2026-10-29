"""Forward decision runner: replay logged decisions through the real engine, decide live on the newest bar.

`run_forward` runs `run_backtest(..., include_last_bar=True)` per mode over snapshot + forward bars,
starting at the first bar >= cfg.start. For every earlier decision bar the `ForwardAgent` replays the
LOGGED weights (no prompt, no model call, so re-running a day never duplicates or changes anything); on
the newest bar it builds the prompt from point-in-time history, asks the pinned model through a
`ForwardCache` bound to (date, mode, model), appends the decision to the log, and returns the orders,
which stay pending until the next open exists. Fills, costs and marking are the historical engine's.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from sab.data.universe import TICKERS
from sab.engine.backtest import BacktestConfig, BacktestResult, Context, run_backtest
from sab.engine.costs import CostModel
from sab.engine.market import MarketData
from sab.engine.types import Order
from sab.forward.cache import ForwardCache, default_cache_dir
from sab.forward.config import FORWARD_DIR, ForwardConfig
from sab.forward.log import DecisionLog
from sab.llm.strategy import LLMAgent
from sab.strategies._base import target_orders

SCHEMA_VERSION = 1
OPEN_UTC = dt.time(13, 30)          # earliest NYSE open (EDT); the decision must be logged before this


class StaleDecision(RuntimeError):
    """The decision window (before the next open) has passed; a late decision proves nothing."""


def decision_deadline(date: str) -> dt.datetime:
    d = dt.date.fromisoformat(date) + dt.timedelta(days=1)
    while d.weekday() >= 5:
        d += dt.timedelta(days=1)
    return dt.datetime.combine(d, OPEN_UTC, tzinfo=dt.timezone.utc)


def _date_str(ts: pd.Timestamp) -> str:
    return str(pd.Timestamp(ts).date())


class ReplayStrategy:
    """Orders from logged weights only. Never calls a model; used for reports and for the replay half
    of the runner."""

    def __init__(self, log: DecisionLog, mode: str, tickers: list[str] | None = None, name: str | None = None) -> None:
        self.log, self.mode = log, mode
        self.tickers = list(tickers) if tickers is not None else list(TICKERS)
        self.name = name or f"forward-{mode}"
        self._by_date = {r["decision_date"]: r for r in log.read() if r["mode"] == mode}

    def on_bar(self, ctx: Context) -> list[Order]:
        rec = self._by_date.get(_date_str(ctx.date))
        return target_orders(ctx, rec["weights"], self.tickers) if rec else []


class ForwardAgent(ReplayStrategy):
    def __init__(self, cfg: ForwardConfig, mode: str, log: DecisionLog, last_index: int, *, directory: Path = FORWARD_DIR,
                 now: Callable[[], dt.datetime] = lambda: dt.datetime.now(dt.timezone.utc), enforce_deadline: bool = True,
                 dry_run: bool = False, tickers: list[str] | None = None) -> None:
        super().__init__(log, mode, tickers, name=f"forward-{mode}")
        self.cfg, self.directory, self.now = cfg, Path(directory), now
        self.last_index, self.enforce_deadline, self.dry_run = last_index, enforce_deadline, dry_run
        self.new_records: list[dict[str, Any]] = []

    def on_bar(self, ctx: Context) -> list[Order]:
        d = _date_str(ctx.date)
        if d in self._by_date:
            return super().on_bar(ctx)
        if ctx.bar_index != self.last_index:
            return []                                    # a missed decision day is a hold, never a backfill
        t = self.now()
        if self.enforce_deadline and t > decision_deadline(d):
            raise StaleDecision(f"{d}/{self.mode}: now {t.isoformat()} is after the deadline {decision_deadline(d).isoformat()}")
        cfg = self.cfg
        agent = LLMAgent(cfg.provider, cfg.model, self.mode, lookback=cfg.lookback, rebalance=cfg.rebalance,
                         temperature=cfg.temperature, seed=cfg.seed, tickers=self.tickers, max_tokens=cfg.max_tokens,
                         cache=ForwardCache(directory=default_cache_dir(self.directory), date=d, mode=self.mode, model=cfg.model))
        agent._initial_equity = cfg.initial_cash
        weights, meta = agent.decide(ctx)
        row, resp = meta["row"], meta["responses"][-1]
        rec = self.log.append({
            "schema": SCHEMA_VERSION, "decision_date": d, "mode": self.mode, "dry_run": self.dry_run,
            "decided_at_utc": self.now().isoformat(timespec="seconds"), "data_last_date": d, "bar_index": int(ctx.bar_index),
            "provider": cfg.provider, "model": cfg.model, "model_served": resp.model, "temperature": cfg.temperature,
            "seed": cfg.seed, "rebalance": cfg.rebalance, "prompt_sha256": row["prompt_sha256"],
            "weights": {k: float(v) for k, v in weights.items() if v > 1e-9}, "exposure": float(sum(weights.values())),
            "rationale": row["rationale"], "raw_response": resp.text, "cache_hit": bool(row["cache_hit"]),
            "retried": bool(row["retried"]), "fallback": bool(row["fallback"]), "repairs": row["repairs"], "error": row["error"],
            "prompt_tokens": int(row["prompt_tokens"]), "completion_tokens": int(row["completion_tokens"]),
        })
        self._by_date[d] = rec
        self.new_records.append(rec)
        return super().on_bar(ctx)


def cost_model(cfg: ForwardConfig) -> CostModel:
    return CostModel(cfg.commission_bps, cfg.slippage_bps)


def first_forward_index(market: MarketData, start: str) -> int:
    return int(market.dates.searchsorted(pd.Timestamp(start)))


def replay_arm(strategy, market: MarketData, cfg: ForwardConfig, rebalance: str | None = None) -> BacktestResult:
    """Run any strategy from the first forward bar (engine rules + costs) and trim to the forward period."""
    first = first_forward_index(market, cfg.start)
    res = run_backtest(strategy, market, BacktestConfig(initial_cash=cfg.initial_cash, rebalance=rebalance or cfg.rebalance,  # type: ignore[arg-type]
                                                        warmup=first, cost_model=cost_model(cfg)), include_last_bar=True)
    start = market.dates[first]
    from dataclasses import replace
    return replace(res, equity=res.equity.loc[start:], cash=res.cash.loc[start:], weights=res.weights.loc[start:],
                   turnover=res.turnover.loc[start:], trades=[f for f in res.trades if f.date >= start])


def run_forward(cfg: ForwardConfig, market: MarketData, *, directory: Path = FORWARD_DIR, modes=("anon", "real"),
                now: Callable[[], dt.datetime] = lambda: dt.datetime.now(dt.timezone.utc), enforce_deadline: bool = True,
                dry_run: bool = False, log_name: str = "decisions.jsonl") -> list[dict[str, Any]]:
    """Decide (if the newest bar is a decision bar and not yet logged) in every mode. Returns the NEW records."""
    directory = Path(directory)
    log = DecisionLog(directory / log_name, min_date=None if dry_run else cfg.start)
    first = first_forward_index(market, cfg.start)
    if first >= len(market):
        return []
    new: list[dict[str, Any]] = []
    for mode in modes:
        agent = ForwardAgent(cfg, mode, log, len(market) - 1, directory=directory, now=now,
                             enforce_deadline=enforce_deadline, dry_run=dry_run)
        replay_arm(agent, market, cfg)
        new.extend(agent.new_records)
    return new
