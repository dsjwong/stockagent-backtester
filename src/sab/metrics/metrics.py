"""Performance metrics from a daily equity curve (marked at close).

Conventions (stated once, used everywhere):
* PERIODS = 252 trading days per year. Years = (n_bars - 1) / 252 (bar count, not calendar).
* Returns are simple daily returns of the equity curve.
* Risk-free rate rf defaults to 0 (annualised, simple); Sharpe/Sortino subtract rf/252 per bar.
  With rf = 0 the Sharpe of cash is 0 and the number is a pure return/vol ratio -- stated
  explicitly in every table header.
* Sharpe uses sample std (ddof=1). Sortino's downside deviation is sqrt(mean(min(r - rf/252, 0)^2)).
* Max drawdown is peak-to-trough on equity; duration is the LONGEST peak-to-recovery
  stretch in bars (an unrecovered drawdown at the end counts up to the last bar).
* Turnover per bar = sum |fill notional| / equity at decision close / 2 (recorded by the
  engine); annualised = total * 252 / n_bars. Buy-and-hold with monthly rebalance is ~0.0x.
* Win rate = share of SELL fills whose fill price exceeds the position's average cost
  (avg cost includes buy-side slippage; commissions excluded on both sides -- an
  approximation, documented). NaN if there are no sells.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from sab.engine.backtest import BacktestResult
from sab.engine.types import Fill

PERIODS = 252


def _returns(equity: pd.Series) -> pd.Series:
    return equity.pct_change().dropna()


def _years(equity: pd.Series) -> float:
    return max(len(equity) - 1, 1) / PERIODS


def cagr(equity: pd.Series) -> float:
    if len(equity) < 2 or equity.iloc[0] <= 0:
        return float("nan")
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1 / _years(equity)) - 1)


def total_return(equity: pd.Series) -> float:
    return float(equity.iloc[-1] / equity.iloc[0] - 1)


def volatility(equity: pd.Series) -> float:
    r = _returns(equity)
    return float(r.std(ddof=1) * math.sqrt(PERIODS)) if len(r) > 1 else float("nan")


def sharpe(equity: pd.Series, rf: float = 0.0) -> float:
    r = _returns(equity) - rf / PERIODS
    sd = r.std(ddof=1)
    if len(r) < 2 or np.isnan(sd) or sd < 1e-12:      # zero-vol curve: undefined, not inf
        return float("nan")
    return float(r.mean() / sd * math.sqrt(PERIODS))


def sortino(equity: pd.Series, rf: float = 0.0) -> float:
    r = _returns(equity) - rf / PERIODS
    downside = np.sqrt(np.mean(np.minimum(r.to_numpy(), 0.0) ** 2))
    if len(r) < 2 or downside < 1e-12:
        return float("nan")
    return float(r.mean() / downside * math.sqrt(PERIODS))


def max_drawdown(equity: pd.Series) -> tuple[float, int]:
    """(max drawdown as a negative fraction, longest drawdown duration in bars)."""
    peak = equity.cummax()
    dd = equity / peak - 1
    mdd = float(dd.min()) if len(dd) else float("nan")
    longest = current = 0
    for v in dd.to_numpy():
        current = current + 1 if v < 0 else 0
        longest = max(longest, current)
    return mdd, longest


def calmar(equity: pd.Series) -> float:
    mdd, _ = max_drawdown(equity)
    return float("nan") if mdd == 0 else cagr(equity) / abs(mdd)


def annualised_turnover(turnover: pd.Series) -> float:
    return float(turnover.sum() * PERIODS / max(len(turnover), 1))


def total_costs(trades: list[Fill]) -> float:
    return float(sum(f.total_cost for f in trades))


def win_rate(trades: list[Fill]) -> float:
    qty: dict[str, int] = {}
    avg: dict[str, float] = {}
    wins = sells = 0
    for f in trades:                       # trades are chronological
        q, a = qty.get(f.ticker, 0), avg.get(f.ticker, 0.0)
        if f.side == "BUY":
            avg[f.ticker] = (a * q + f.price * f.qty) / (q + f.qty)
            qty[f.ticker] = q + f.qty
        else:
            sells += 1
            wins += f.price > a
            qty[f.ticker] = q - f.qty
    return wins / sells if sells else float("nan")


def compute_metrics(res: BacktestResult, rf: float = 0.0) -> dict[str, Any]:
    eq = res.equity
    mdd, dur = max_drawdown(eq)
    return {
        "CAGR": cagr(eq),
        "TotalRet": total_return(eq),
        "Vol": volatility(eq),
        "Sharpe": sharpe(eq, rf),
        "Sortino": sortino(eq, rf),
        "MaxDD": mdd,
        "DD_days": dur,
        "Calmar": calmar(eq),
        "Turnover": annualised_turnover(res.turnover),
        "Exposure": float(res.exposure.mean()),
        "WinRate": win_rate(res.trades),
        "Trades": len(res.trades),
        "Costs": total_costs(res.trades),
        "Costs_bps": total_costs(res.trades) / res.equity.iloc[0] * 1e4,
    }


def summary_table(results: dict[str, BacktestResult], rf: float = 0.0) -> pd.DataFrame:
    """One row per named result; columns = metrics (Sharpe/Sortino at the given rf)."""
    rows = {name: compute_metrics(r, rf) for name, r in results.items()}
    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = f"strategy (rf={rf:g})"
    return df


def format_table(df: pd.DataFrame) -> str:
    """Markdown table with sensible formatting per column."""
    pct = {"CAGR", "TotalRet", "Vol", "MaxDD", "Exposure", "WinRate"}
    out = df.copy()
    for c in out.columns:
        if c in pct:
            out[c] = out[c].map(lambda v: "" if pd.isna(v) else f"{v:.1%}")
        elif c in {"Sharpe", "Sortino", "Calmar", "Turnover"}:
            out[c] = out[c].map(lambda v: "" if pd.isna(v) else f"{v:.2f}")
        elif c in {"Costs"}:
            out[c] = out[c].map(lambda v: f"{v:,.0f}")
        elif c in {"Costs_bps"}:
            out[c] = out[c].map(lambda v: f"{v:.0f}")
    return _to_markdown(out)


def _to_markdown(df: pd.DataFrame) -> str:
    """Tiny markdown renderer (avoids the `tabulate` dependency)."""
    cols = [str(df.index.name or "")] + [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for idx, row in df.iterrows():
        lines.append("| " + " | ".join([str(idx)] + [str(v) for v in row.tolist()]) + " |")
    return "\n".join(lines)
