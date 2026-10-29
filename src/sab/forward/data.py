"""Incremental price fetch for the forward test, and the combined (snapshot + forward) market.

The committed historical snapshots are never modified. New bars go to forward/prices/<T>.parquet,
APPEND-ONLY: a bar is stored only if its date is later than every stored date, and only if the
bar is complete (US close has passed). A partial intraday bar is the one lookahead trap a live
fetch has, so `latest_complete_bar_date` caps what may be stored. Prices are as fetched on the day
(point-in-time, not re-adjusted later): a dividend shows as an ex-date price drop for every arm alike.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd

from sab.data.download import COLUMNS, fetch_yfinance
from sab.data.loader import load_ohlcv
from sab.data.universe import ALL_SYMBOLS, TICKERS
from sab.engine.market import MarketData
from sab.forward.config import FORWARD_DIR

CLOSE_CUTOFF_UTC = dt.time(21, 30)      # NYSE closes 20:00 UTC (EDT) / 21:00 UTC (EST); +30 min margin

Fetcher = Callable[[str, str, str], pd.DataFrame]


class ForwardDataError(RuntimeError):
    pass


def _weekday_on_or_before(d: dt.date) -> dt.date:
    while d.weekday() >= 5:
        d -= dt.timedelta(days=1)
    return d


def latest_complete_bar_date(now: dt.datetime) -> dt.date:
    """Newest date whose daily bar can be final at `now` (UTC): today once the cutoff has passed,
    else the previous weekday. Exchange holidays are handled by the data simply having no row."""
    now = now.astimezone(dt.timezone.utc) if now.tzinfo else now.replace(tzinfo=dt.timezone.utc)
    today = now.date()
    if now.time() >= CLOSE_CUTOFF_UTC and today.weekday() < 5:
        return today
    return _weekday_on_or_before(today - dt.timedelta(days=1))


def price_path(ticker: str, directory: Path = FORWARD_DIR) -> Path:
    return Path(directory) / "prices" / f"{ticker}.parquet"


def load_forward_bars(ticker: str, directory: Path = FORWARD_DIR) -> pd.DataFrame | None:
    p = price_path(ticker, directory)
    if not p.exists():
        return None
    df = pd.read_parquet(p, engine="pyarrow")
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"
    return df


def last_known_date(ticker: str, directory: Path = FORWARD_DIR, snapshot_loader=load_ohlcv) -> pd.Timestamp:
    fwd = load_forward_bars(ticker, directory)
    if fwd is not None and len(fwd):
        return fwd.index[-1]
    return snapshot_loader(ticker).index[-1]


def fetch_incremental(now: dt.datetime, directory: Path = FORWARD_DIR, fetcher: Fetcher = fetch_yfinance,
                      symbols: Iterable[str] = ALL_SYMBOLS, snapshot_loader=load_ohlcv) -> dict[str, int]:
    """Append complete bars after each symbol's last known date. Returns {symbol: n_new_bars}.
    Existing rows are never touched. Raises ForwardDataError if the symbols end on different dates
    afterwards (a half-updated universe must not be decided on)."""
    cutoff = pd.Timestamp(latest_complete_bar_date(now))
    added: dict[str, int] = {}
    for t in symbols:
        last = last_known_date(t, directory, snapshot_loader)
        if last >= cutoff:
            added[t] = 0
            continue
        start = (last + pd.Timedelta(days=1)).date().isoformat()
        end = (cutoff + pd.Timedelta(days=1)).date().isoformat()        # yfinance end is exclusive
        try:
            new = fetcher(t, start, end)
        except RuntimeError as e:                                         # "no rows" on a holiday window
            if "no rows" in str(e):
                added[t] = 0
                continue
            raise ForwardDataError(f"{t}: fetch failed: {e}") from e
        new = new.loc[(new.index > last) & (new.index <= cutoff), COLUMNS]
        if new.empty:
            added[t] = 0
            continue
        old = load_forward_bars(t, directory)
        out = new if old is None else pd.concat([old, new])
        if out.index.has_duplicates or not out.index.is_monotonic_increasing:
            raise ForwardDataError(f"{t}: appended bars would break date order")
        p = price_path(t, directory)
        p.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(p, engine="pyarrow")
        added[t] = int(len(new))
    ends = {t: last_known_date(t, directory, snapshot_loader) for t in symbols}
    if len(set(ends.values())) > 1:
        raise ForwardDataError("symbols end on different dates after fetch: "
                               + ", ".join(f"{t}={d.date()}" for t, d in sorted(ends.items())))
    return added


def load_forward_market(directory: Path = FORWARD_DIR, tickers: Iterable[str] | None = None,
                        snapshot_loader=load_ohlcv) -> MarketData:
    """Snapshot + forward bars, adjusted open/close like `load_market`. Default columns: universe + SPY."""
    cols = list(tickers) if tickers is not None else list(TICKERS) + ["SPY"]
    opens, closes = {}, {}
    for t in cols:
        df = snapshot_loader(t)
        fwd = load_forward_bars(t, directory)
        if fwd is not None and len(fwd):
            fwd = fwd.loc[fwd.index > df.index[-1]]
            df = pd.concat([df, fwd[COLUMNS]])
        f = df["Adj Close"] / df["Close"]
        opens[t] = (df["Open"] * f).rename(t)
        closes[t] = df["Adj Close"].rename(t)
    return MarketData(pd.concat(opens.values(), axis=1, join="inner").sort_index(),
                      pd.concat(closes.values(), axis=1, join="inner").sort_index())
