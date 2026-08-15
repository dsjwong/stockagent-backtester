"""Download daily OHLCV snapshots (the ONLY module in sab that touches the network).

Usage:
    python -m sab.data.download            # download tickers missing a snapshot
    python -m sab.data.download --force    # re-download everything

Writes one parquet per symbol to data/snapshots/{TICKER}.parquet with columns
    Open, High, Low, Close, Adj Close, Volume        (index: DatetimeIndex 'Date')
using yfinance with auto_adjust=False so BOTH the raw Close and the split/dividend
adjusted 'Adj Close' are kept (see docs/decisions.md). Falls back to Stooq
(https://stooq.com/q/d/l/?s=aapl.us&i=d) if yfinance returns nothing; Stooq bars are
already split-adjusted but NOT dividend-adjusted, so 'Adj Close' is set equal to
'Close' and the manifest records source="stooq" so the difference is never hidden.

data/snapshots/MANIFEST.json records per ticker: rows, first/last date, source,
download timestamp, yfinance version -- what makes a snapshot reproducible/auditable.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import io
import json
import time
from pathlib import Path

import pandas as pd
import requests

from sab.data.universe import ALL_SYMBOLS, START, end_date

SNAPSHOT_DIR = Path(__file__).resolve().parents[3] / "data" / "snapshots"
MANIFEST_PATH = SNAPSHOT_DIR / "MANIFEST.json"
COLUMNS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]


def _tidy(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise to COLUMNS, tz-naive DatetimeIndex named 'Date', sorted, de-duplicated."""
    if isinstance(df.columns, pd.MultiIndex):        # yfinance>=0.2.40 returns (field, ticker)
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    df = df.loc[:, [c for c in COLUMNS if c in df.columns]].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "Date"
    df.columns.name = None
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.dropna(subset=["Close"])
    for c in COLUMNS:
        if c not in df.columns:
            raise ValueError(f"missing column {c!r} after tidy: {list(df.columns)}")
    return df[COLUMNS].astype({"Volume": "float64"})


def fetch_yfinance(ticker: str, start: str, end: str) -> pd.DataFrame:
    import yfinance as yf

    df = yf.download(ticker, start=start, end=end, auto_adjust=False,
                     progress=False, threads=False)
    if df is None or df.empty:
        raise RuntimeError(f"yfinance returned no rows for {ticker}")
    return _tidy(df)


def fetch_stooq(ticker: str, start: str, end: str) -> pd.DataFrame:
    url = f"https://stooq.com/q/d/l/?s={ticker.lower()}.us&i=d"
    r = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    head = r.text[:300].lower()
    if r.status_code != 200 or "no data" in head:
        raise RuntimeError(f"stooq returned status {r.status_code} / no data for {ticker}")
    if "<html" in head or "requires javascript" in head:
        # As of 2026-08 stooq fronts its CSV endpoint with a JavaScript proof-of-work
        # challenge, so plain HTTP clients get an HTML page instead of CSV.
        raise RuntimeError(f"stooq served a JS-verification page instead of CSV for {ticker}")
    if not head.startswith("date,"):
        raise RuntimeError(f"stooq returned unexpected content for {ticker}: {r.text[:60]!r}")
    df = pd.read_csv(io.StringIO(r.text), parse_dates=["Date"], index_col="Date")
    df["Adj Close"] = df["Close"]          # stooq: split-adjusted only, no dividend adjustment
    df = _tidy(df)
    return df.loc[start:end]


def _load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text())
    return {"tickers": {}}


def download_one(ticker: str, start: str, end: str) -> tuple[pd.DataFrame, str]:
    """Return (frame, source). Tries yfinance, then Stooq. Raises if both fail."""
    errors = []
    try:
        return fetch_yfinance(ticker, start, end), "yfinance"
    except Exception as e:  # noqa: BLE001 - we want to fall through to stooq on anything
        errors.append(f"yfinance: {e}")
    try:
        return fetch_stooq(ticker, start, end), "stooq"
    except Exception as e:  # noqa: BLE001
        errors.append(f"stooq: {e}")
    raise RuntimeError(f"{ticker}: all sources failed -> " + " | ".join(errors))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--force", action="store_true", help="re-download even if a snapshot exists")
    ap.add_argument("--tickers", nargs="*", default=None, help="subset of symbols (default: universe + SPY)")
    ap.add_argument("--start", default=START)
    ap.add_argument("--end", default=None, help="ISO date; default = today")
    args = ap.parse_args(argv)

    import yfinance as yf

    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest()
    end = args.end or end_date()
    symbols = args.tickers or list(ALL_SYMBOLS)
    failures = []

    for t in symbols:
        path = SNAPSHOT_DIR / f"{t}.parquet"
        if path.exists() and not args.force and t in manifest["tickers"]:
            print(f"[skip] {t}: snapshot exists ({manifest['tickers'][t]['rows']} rows)")
            continue
        try:
            df, source = download_one(t, args.start, end)
        except RuntimeError as e:
            print(f"[FAIL] {e}")
            failures.append(t)
            continue
        df.to_parquet(path, engine="pyarrow")
        manifest["tickers"][t] = {
            "rows": int(len(df)),
            "first_date": df.index[0].date().isoformat(),
            "last_date": df.index[-1].date().isoformat(),
            "source": source,
            "downloaded_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
            "yfinance_version": yf.__version__,
            "requested_start": args.start,
            "requested_end": end,
        }
        print(f"[ok]   {t}: {len(df)} rows {df.index[0].date()} -> {df.index[-1].date()} ({source})")
        time.sleep(0.5)   # be polite to free endpoints
        MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    manifest["generated_at"] = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    if failures:
        print(f"FAILED: {failures}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
