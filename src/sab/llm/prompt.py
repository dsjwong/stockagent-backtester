"""Prompt builder for the LLM agent, and the no-future audit helper.

The prompt is the ONLY thing the model sees, so it is built from a price frame that already
ends at the decision bar (`ctx.prices(...)`, i.e. the engine's `iloc[:t+1]` view) plus the
portfolio state. It contains, in order:

  1. role + rules (long-only, weights >= 0, sum <= 1, remainder is cash, costs per side),
  2. `Tickers: ...` (the machine-readable universe line the mock providers read),
  3. a compact price table: daily closes for the last 10 bars, then every 5th bar back to
     `lookback` bars, oldest first,
  4. trailing 1w / 1m / 3m / 6m returns per ticker (5 / 21 / 63 / 126 bars; "n/a" if the history
     handed in is shorter),
  5. portfolio: current weights, cash %, equity vs start,
  6. strict output spec: JSON {"weights": {...}, "rationale": "<= 40 words"}.

Modes
* `real`: tickers are their symbols, rows are calendar dates, prices are adjusted closes.
* `anon`: tickers become `Stock 01..NN` through a seed-shuffled bijection (stored in the run
  config so decisions can be de-anonymised), rows are `Day <t>` (bar index), and prices are
  rebased to 100 at the first row of the window so levels do not leak the identity either.
  Purpose: measure the model without whatever it remembers about AAPL in 2022.

`assert_prompt_no_future(prompt, ctx_last_date, full_frame)` is the audit: it formats every
price value strictly after `ctx_last_date` the way the prompt formats prices and asserts none
appears in the prompt (values that also occur inside the visible window are excluded, since a
past close can legitimately equal a future one), and in real mode also that no future date
string appears. Tests run it on real data at several dates; the engine-level perturbation
harness is the second, independent proof.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

import numpy as np
import pandas as pd

Mode = Literal["real", "anon"]
RETURN_WINDOWS = {"1w": 5, "1m": 21, "3m": 63, "6m": 126}
DAILY_ROWS = 10
WEEKLY_STEP = 5
PRICE_FMT = "{:.2f}"


@dataclass(frozen=True)
class AnonMapping:
    """Bijection ticker -> 'Stock NN', shuffled by `seed`. Round-trips exactly."""

    to_anon: dict[str, str]
    to_real: dict[str, str]
    seed: int

    @classmethod
    def make(cls, tickers: list[str], seed: int = 0) -> "AnonMapping":
        tickers = list(tickers)
        if len(set(tickers)) != len(tickers):
            raise ValueError("tickers must be unique")
        order = np.random.default_rng(seed).permutation(len(tickers))
        width = max(2, len(str(len(tickers))))
        to_anon = {tickers[i]: f"Stock {j + 1:0{width}d}" for j, i in enumerate(order)}
        return cls(to_anon=to_anon, to_real={v: k for k, v in to_anon.items()}, seed=int(seed))

    @classmethod
    def identity(cls, tickers: list[str]) -> "AnonMapping":
        return cls(to_anon={t: t for t in tickers}, to_real={t: t for t in tickers}, seed=-1)

    def anon(self, ticker: str) -> str:
        return self.to_anon[ticker]

    def real(self, name: str) -> str:
        return self.to_real[name]

    def to_dict(self) -> dict[str, Any]:
        return {"seed": self.seed, "ticker_to_anon": dict(self.to_anon)}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "AnonMapping":
        ta = dict(d["ticker_to_anon"])
        return cls(to_anon=ta, to_real={v: k for k, v in ta.items()}, seed=int(d.get("seed", -1)))


@dataclass
class PromptSpec:
    """A built prompt plus what is needed to interpret the answer."""

    text: str
    mode: Mode
    tickers: list[str]                    # real tickers, in prompt order
    names: list[str]                      # how they are called in the prompt (== tickers in real mode)
    mapping: AnonMapping
    last_date: pd.Timestamp
    day_index: int
    lookback: int
    n_rows: int
    meta: dict[str, Any] = field(default_factory=dict)

    def real_ticker(self, name: str) -> str | None:
        return self.mapping.to_real.get(name)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- pieces

def _rebase(px: pd.DataFrame) -> pd.DataFrame:
    """Rebase every column to 100 at the first row (anon mode)."""
    return px / px.iloc[0] * 100.0


def table_rows(n: int, daily: int = DAILY_ROWS, step: int = WEEKLY_STEP) -> list[int]:
    """Row positions (0..n-1) shown in the price table: last `daily` bars, then every `step`
    bars further back, oldest first."""
    if n <= 0:
        return []
    last_daily = list(range(max(0, n - daily), n))
    weekly = list(range(last_daily[0] - step, -1, -step))
    return sorted(set(weekly + last_daily))


def _fmt_price(v: float) -> str:
    return PRICE_FMT.format(v)


def _fmt_ret(v: float) -> str:
    return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v * 100:+.1f}%"


def _row_label(mode: Mode, date: pd.Timestamp, day_index: int) -> str:
    return f"Day {day_index}" if mode == "anon" else str(pd.Timestamp(date).date())


def _price_table(px: pd.DataFrame, names: list[str], mode: Mode, day_index_last: int) -> str:
    n = len(px)
    header = "date" if mode == "real" else "day"
    lines = [" | ".join([header] + names)]
    for i in table_rows(n):
        label = _row_label(mode, px.index[i], day_index_last - (n - 1 - i))
        lines.append(" | ".join([label] + [_fmt_price(v) for v in px.iloc[i].to_numpy()]))
    return "\n".join(lines)


def _returns_block(full: pd.DataFrame, names: list[str]) -> str:
    """Trailing returns from the full history handed in (not just the table window)."""
    lines = ["ticker | " + " | ".join(RETURN_WINDOWS)]
    last = full.iloc[-1]
    for t, name in zip(full.columns, names):
        vals = [_fmt_ret(float(last[t] / full[t].iloc[-1 - w] - 1.0) if len(full) > w else float("nan"))
                for w in RETURN_WINDOWS.values()]
        lines.append(f"{name} | " + " | ".join(vals))
    return "\n".join(lines)


def _portfolio_block(portfolio_state: Mapping[str, Any], tickers: list[str], names: list[str]) -> str:
    w = portfolio_state.get("weights", {}) or {}
    equity = float(portfolio_state.get("equity", 0.0))
    cash = float(portfolio_state.get("cash", equity))
    start = float(portfolio_state.get("initial_equity", equity) or equity)
    held = [f"{name} {float(w.get(t, 0.0)):.1%}" for t, name in zip(tickers, names) if float(w.get(t, 0.0)) > 0.0005]
    cash_pct = cash / equity if equity > 0 else 1.0
    pnl = equity / start - 1.0 if start > 0 else 0.0
    return "\n".join([
        f"Current weights: {', '.join(held) if held else 'none (all cash)'}",
        f"Cash: {cash_pct:.1%} of equity",
        f"Equity vs start: {pnl:+.1%}",
    ])


RULES = (
    "You are a portfolio manager deciding target weights for a long-only stock portfolio.\n"
    "Rules: weights are fractions of total equity, each in [0, 1], the sum must be <= 1; whatever is not "
    "allocated stays in cash. No shorting, no leverage. Orders fill at the next open and cost about 0.15% per "
    "side, so do not churn without reason. Decide from the data below only."
)


def output_spec(names: list[str]) -> str:
    ex = {n: 0.0 for n in names[:2]}
    return (
        "Respond with ONLY a JSON object, no prose before or after, exactly of the form\n"
        '{"weights": {"<ticker>": <fraction>, ...}, "rationale": "<= 40 words"}\n'
        f"using only these tickers ({', '.join(names)}). Omitted tickers get weight 0. Example shape: "
        f"{json.dumps({'weights': ex, 'rationale': '...'})}"
    )


# --------------------------------------------------------------------------- builder

def build_prompt(ctx_prices: pd.DataFrame, portfolio_state: Mapping[str, Any], mode: Mode = "real",
                 lookback: int = 60, *, mapping: AnonMapping | None = None, day_index: int | None = None,
                 tickers: list[str] | None = None) -> PromptSpec:
    """Build the decision prompt from a price frame that ENDS at the decision bar.

    ctx_prices      : adjusted closes, index = dates, columns = tickers, last row = decision bar.
                      Rows beyond `lookback` are used only for the trailing-return block.
    portfolio_state : {"weights": {t: w}, "cash": $, "equity": $, "initial_equity": $}
    mode            : "real" | "anon"
    mapping         : AnonMapping for anon mode (made from the tickers with seed 0 if omitted)
    day_index       : bar index of the last row (anon row labels); default len(ctx_prices)-1
    tickers         : subset/order of columns to include (default: all columns)
    """
    if mode not in ("real", "anon"):
        raise ValueError("mode must be 'real' or 'anon'")
    if len(ctx_prices) == 0:
        raise ValueError("ctx_prices is empty")
    cols = list(tickers) if tickers is not None else list(ctx_prices.columns)
    full = ctx_prices[cols]
    if day_index is None:
        day_index = len(full) - 1
    if mapping is None:
        mapping = AnonMapping.make(cols, seed=0) if mode == "anon" else AnonMapping.identity(cols)
    if mode == "anon":                       # present in Stock 01..NN order so column order leaks nothing
        cols = sorted(cols, key=lambda t: mapping.to_anon[t])
        full = full[cols]
    names = [mapping.to_anon[t] for t in cols] if mode == "anon" else list(cols)

    window = full.iloc[-lookback:]
    if mode == "anon":
        window = _rebase(window)
    last_date = pd.Timestamp(full.index[-1])
    header_date = f"Today: {last_date.date()}" if mode == "real" else f"Today: Day {day_index}"
    price_note = ("Adjusted closes." if mode == "real"
                  else f"Prices rebased to 100 at Day {day_index - len(window) + 1}; identities are hidden on purpose.")
    parts = [
        RULES,
        header_date,
        f"Tickers: {', '.join(names)}",
        "",
        f"Recent closes (last {min(DAILY_ROWS, len(window))} bars daily, then every {WEEKLY_STEP} bars, oldest first). {price_note}",
        _price_table(window, names, mode, day_index),
        "",
        "Trailing returns:",
        _returns_block(full, names),
        "",
        "Portfolio:",
        _portfolio_block(portfolio_state, cols, names),
        "",
        output_spec(names),
    ]
    text = "\n".join(parts)
    return PromptSpec(text=text, mode=mode, tickers=cols, names=names, mapping=mapping, last_date=last_date,
                      day_index=int(day_index), lookback=int(lookback), n_rows=len(table_rows(len(window))),
                      meta={"n_hist_rows": len(full)})


# --------------------------------------------------------------------------- audit helper

def _price_pattern(s: str) -> re.Pattern:
    return re.compile(r"(?<![\d.,])" + re.escape(s) + r"(?![\d.,%])")


def assert_prompt_no_future(prompt: str, ctx_last_date, full_frame: pd.DataFrame, *, mode: Mode = "real",
                            lookback: int = 60, tickers: list[str] | None = None) -> dict[str, int]:
    """Raise AssertionError if any formatted price value from bars strictly after `ctx_last_date`
    (or, in real mode, any such date string) appears in `prompt`. Values that also occur in the
    visible past window are skipped (a past close may equal a future one to 2 dp). Returns counts
    for reporting. `mode`/`lookback` reproduce the prompt's own price transformation for anon."""
    last = pd.Timestamp(ctx_last_date)
    cols = list(tickers) if tickers is not None else list(full_frame.columns)
    frame = full_frame[cols]
    past = frame.loc[:last]
    future = frame.loc[frame.index > last]
    if future.empty:
        return {"future_values": 0, "leaks": 0}
    if mode == "anon":
        base = past.iloc[-lookback:].iloc[0]
        past_v = past.iloc[-lookback:] / base * 100.0
        future_v = future / base * 100.0
    else:
        past_v, future_v = past.iloc[-max(lookback, RETURN_WINDOWS["6m"] + 1):], future
    seen = {_fmt_price(v) for v in past_v.to_numpy().ravel()}
    fut_vals = {_fmt_price(v) for v in future_v.to_numpy().ravel()} - seen
    leaks = [v for v in fut_vals if _price_pattern(v).search(prompt)]
    assert not leaks, f"future price values found in prompt: {sorted(leaks)[:10]}"
    if mode == "real":
        leaked_dates = [d for d in (str(pd.Timestamp(d).date()) for d in future.index) if d in prompt]
        assert not leaked_dates, f"future dates found in prompt: {leaked_dates[:5]}"
    return {"future_values": len(fut_vals), "leaks": 0}
