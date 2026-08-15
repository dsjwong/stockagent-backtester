"""Offline price loader. Reads ONLY data/snapshots/*.parquet -- never touches the network.

The backtest engine (M2+) must import prices exclusively through this module, so a
backtest run is reproducible from the committed snapshot and costs zero network calls.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd

from sab.data.universe import TICKERS

SNAPSHOT_DIR = Path(__file__).resolve().parents[3] / "data" / "snapshots"
FIELDS = ("Open", "High", "Low", "Close", "Adj Close", "Volume")
MAX_GAP_BDAYS = 5


class SnapshotMissingError(FileNotFoundError):
    pass


def _snapshot_path(ticker: str) -> Path:
    p = SNAPSHOT_DIR / f"{ticker}.parquet"
    if not p.exists():
        raise SnapshotMissingError(
            f"No snapshot for {ticker!r} at {p}. Run `python -m sab.data.download` first "
            "(the loader never downloads)."
        )
    return p


def _validate(df: pd.DataFrame, ticker: str) -> None:
    idx = df.index
    if not isinstance(idx, pd.DatetimeIndex):
        raise ValueError(f"{ticker}: index is not a DatetimeIndex")
    if idx.has_duplicates:
        raise ValueError(f"{ticker}: duplicate dates in snapshot")
    if not idx.is_monotonic_increasing:
        raise ValueError(f"{ticker}: dates not monotonic increasing")
    # gaps > MAX_GAP_BDAYS business days are flagged (warning, not error): could be a
    # data hole or a real halt; either way the user should know.
    bdays = pd.Series(idx[1:]).sub(pd.Series(idx[:-1])).dt.days
    # 5 business days ~ 7 calendar days; allow long weekends by using calendar days > 9
    gaps = bdays[bdays > 9]
    if len(gaps):
        where = idx[1:][gaps.index]
        warnings.warn(f"{ticker}: {len(gaps)} gap(s) > {MAX_GAP_BDAYS} business days, "
                      f"e.g. before {where[0].date()}", stacklevel=3)


def load_ohlcv(ticker: str) -> pd.DataFrame:
    """Full OHLCV frame for one ticker (Open, High, Low, Close, Adj Close, Volume)."""
    df = pd.read_parquet(_snapshot_path(ticker), engine="pyarrow")
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"
    _validate(df, ticker)
    return df


def load_prices(tickers=None, start=None, end=None, field: str = "Adj Close") -> pd.DataFrame:
    """Wide frame: index = dates, columns = tickers, values = `field`.

    Rows are the INTERSECTION of dates across tickers (so a bar exists for every
    ticker), sliced to [start, end]. Default tickers = the fixed universe (SPY excluded;
    ask for it explicitly).
    """
    if field not in FIELDS:
        raise ValueError(f"field must be one of {FIELDS}, got {field!r}")
    tickers = list(tickers) if tickers is not None else list(TICKERS)
    cols = {t: load_ohlcv(t)[field].rename(t) for t in tickers}
    wide = pd.concat(cols.values(), axis=1, join="inner").sort_index()
    if start is not None:
        wide = wide.loc[pd.Timestamp(start):]
    if end is not None:
        wide = wide.loc[:pd.Timestamp(end)]
    if wide.empty:
        raise ValueError("no overlapping dates for the requested tickers/window")
    return wide
