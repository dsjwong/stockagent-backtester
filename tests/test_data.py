"""M1 data-layer tests. Run: .venv/bin/pytest -q  (needs data/snapshots/, committed)."""

import json
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sab.data import loader
from sab.data.universe import ALL_SYMBOLS, BENCHMARK, START, TICKERS

SNAP = Path(__file__).resolve().parents[1] / "data" / "snapshots"


def test_universe_shape():
    assert len(TICKERS) == 15
    assert len(set(TICKERS)) == 15
    assert BENCHMARK not in TICKERS


def test_manifest_exists_and_covers_universe():
    m = json.loads((SNAP / "MANIFEST.json").read_text())
    for t in ALL_SYMBOLS:
        assert t in m["tickers"], f"{t} missing from manifest"
        entry = m["tickers"][t]
        assert entry["rows"] > 1500                       # ~8 years of daily bars
        assert entry["first_date"] <= "2018-01-03"
        assert entry["source"] in {"yfinance", "stooq"}
        assert (SNAP / f"{t}.parquet").exists()


def test_load_ohlcv_columns_and_index():
    df = loader.load_ohlcv("AAPL")
    assert list(df.columns) == list(loader.FIELDS)
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.is_monotonic_increasing and not df.index.has_duplicates
    assert df.index[0] >= pd.Timestamp(START)
    # adjusted close differs from close for a dividend payer (proves auto_adjust=False kept both)
    assert not np.allclose(df["Close"].iloc[:250], df["Adj Close"].iloc[:250])


def test_load_prices_aligned_positive_no_nan():
    with warnings.catch_warnings():
        warnings.simplefilter("error")          # a >5-bday gap in large-caps would be a data bug
        px = loader.load_prices()
    assert list(px.columns) == list(TICKERS)
    assert px.notna().all().all()
    assert (px > 0).all().all()
    assert px.index.is_monotonic_increasing and not px.index.has_duplicates
    assert len(px) > 1500


def test_load_prices_window_and_field():
    px = loader.load_prices(["AAPL", BENCHMARK], start="2022-01-01", end="2022-12-31", field="Close")
    assert px.index[0].year == 2022 and px.index[-1].year == 2022
    assert list(px.columns) == ["AAPL", BENCHMARK]
    assert 245 <= len(px) <= 255                        # ~251 trading days in 2022


def test_missing_snapshot_raises_clearly(tmp_path, monkeypatch):
    monkeypatch.setattr(loader, "SNAPSHOT_DIR", tmp_path)
    with pytest.raises(loader.SnapshotMissingError, match="sab.data.download"):
        loader.load_ohlcv("AAPL")


def test_loader_does_not_import_network_code():
    # fresh interpreter: importing the loader and reading prices must not pull in
    # yfinance/requests (structural guarantee that backtests are offline).
    code = (
        "import sys; from sab.data import loader; loader.load_prices(['AAPL']);"
        "bad=[m for m in ('yfinance','requests','urllib3') if m in sys.modules];"
        "print(','.join(bad))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "", f"loader imported network modules: {out.stdout}"
