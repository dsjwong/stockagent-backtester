"""Forward-test tests. Offline: mock provider, synthetic prices, temp directories, injected clocks and fetchers."""

from __future__ import annotations

import datetime as dt
import json

import numpy as np
import pandas as pd
import pytest

from sab.data.download import COLUMNS
from sab.data.universe import TICKERS
from sab.engine import BacktestConfig, run_backtest
from sab.forward import (DecisionLog, DuplicateDecision, ForwardCache, ForwardConfig, ForwardDataError, LogIntegrityError,
                         StaleDecision, decision_deadline, fetch_incremental, forward_cache_key, latest_complete_bar_date,
                         run_forward)
from sab.forward.analysis import MIN_DAYS_FOR_CI, bootstrap_gap
from sab.forward.log import GENESIS
from sab.llm import complete
from sab.strategies import EqualWeight
from tests.test_llm import rw_market

UTC = dt.timezone.utc


# ------------------------------------------------------------------ cache isolation

def test_forward_cache_key_depends_on_date_mode_model():
    base = "abc"
    keys = {forward_cache_key(d, m, mod, base) for d in ("2026-11-02", "2026-11-03") for m in ("anon", "real") for mod in ("m1", "m2")}
    assert len(keys) == 8


def test_forward_cache_never_returns_other_date_mode_or_model(tmp_path):
    prompt = "Tickers: A, B\nsame prompt"
    caches = {}
    for date, mode, model in [("2026-11-02", "anon", "mock_ew"), ("2026-11-03", "anon", "mock_ew"),
                              ("2026-11-02", "real", "mock_ew")]:
        caches[(date, mode, model)] = ForwardCache(directory=tmp_path, date=date, mode=mode, model=model)
    a = caches[("2026-11-02", "anon", "mock_ew")]
    r1 = complete("mock", "mock_ew", prompt, cache=a)
    assert not r1.cached
    assert complete("mock", "mock_ew", prompt, cache=a).cached
    for k, c in caches.items():                           # identical prompt, other binding: must MISS
        if k != ("2026-11-02", "anon", "mock_ew"):
            assert not complete("mock", "mock_ew", prompt, cache=c).cached
    assert len(list(tmp_path.glob("*.json"))) == 3


def test_forward_cache_rejects_record_with_foreign_binding(tmp_path):
    a = ForwardCache(directory=tmp_path, date="2026-11-02", mode="anon", model="m")
    b = ForwardCache(directory=tmp_path, date="2026-11-03", mode="anon", model="m")
    a.put("k", {"response": {"text": "old"}})
    b.path("k").write_text(a.path("k").read_text())      # stale file planted under b's filename
    assert b.get("k") is None and b.hits == 0


# ------------------------------------------------------------------ clock / fetch

def test_latest_complete_bar_date():
    assert latest_complete_bar_date(dt.datetime(2026, 10, 29, 15, 0, tzinfo=UTC)) == dt.date(2026, 10, 28)   # Thu intraday
    assert latest_complete_bar_date(dt.datetime(2026, 10, 29, 22, 30, tzinfo=UTC)) == dt.date(2026, 10, 29)  # Thu after close
    assert latest_complete_bar_date(dt.datetime(2026, 10, 31, 12, 0, tzinfo=UTC)) == dt.date(2026, 10, 30)   # Sat -> Fri
    assert latest_complete_bar_date(dt.datetime(2026, 11, 2, 9, 0, tzinfo=UTC)) == dt.date(2026, 10, 30)     # Mon pre-open -> Fri


def _frame(dates, base=100.0):
    idx = pd.DatetimeIndex(pd.to_datetime(dates), name="Date")
    v = base + np.arange(len(idx), dtype=float)
    return pd.DataFrame({"Open": v, "High": v + 1, "Low": v - 1, "Close": v, "Adj Close": v, "Volume": 1e6}, index=idx)[COLUMNS]


def _snap(ticker):
    return _frame(["2026-10-22", "2026-10-23"])


def test_fetch_never_stores_future_or_partial_bars_and_never_edits_old_rows(tmp_path):
    asked = []

    def fetcher(t, start, end):
        asked.append((t, start, end))
        # returns: an already-known date, valid new dates, today's partial bar and a future bar
        return _frame(["2026-10-23", "2026-10-26", "2026-10-27", "2026-10-28", "2026-10-29", "2026-10-30"], base=500.0)

    now = dt.datetime(2026, 10, 28, 15, 0, tzinfo=UTC)     # Wed intraday: the 28th is NOT complete yet
    added = fetch_incremental(now, tmp_path, fetcher, symbols=["AAA", "BBB"], snapshot_loader=_snap)
    assert added == {"AAA": 2, "BBB": 2}
    stored = pd.read_parquet(tmp_path / "prices" / "AAA.parquet")
    assert stored.index.max() == pd.Timestamp("2026-10-27")
    assert pd.Timestamp("2026-10-23") not in stored.index                          # snapshot rows are never re-written
    before = stored.copy()
    now2 = dt.datetime(2026, 10, 28, 22, 30, tzinfo=UTC)   # after the close: the 28th may be stored, the 29th/30th may not
    fetch_incremental(now2, tmp_path, fetcher, symbols=["AAA", "BBB"], snapshot_loader=_snap)
    after = pd.read_parquet(tmp_path / "prices" / "AAA.parquet")
    assert after.index.max() == pd.Timestamp("2026-10-28")
    pd.testing.assert_frame_equal(after.loc[before.index], before)                 # old rows untouched
    assert fetch_incremental(now2, tmp_path, fetcher, symbols=["AAA", "BBB"], snapshot_loader=_snap) == {"AAA": 0, "BBB": 0}


def test_fetch_refuses_half_updated_universe(tmp_path):
    def fetcher(t, start, end):
        return _frame(["2026-10-26"]) if t == "AAA" else _frame(["2026-10-26"]).iloc[0:0]

    with pytest.raises(ForwardDataError):
        fetch_incremental(dt.datetime(2026, 10, 27, 12, 0, tzinfo=UTC), tmp_path, fetcher, symbols=["AAA", "BBB"], snapshot_loader=_snap)


# ------------------------------------------------------------------ append-only log

def _rec(date, mode="anon", **kw):
    return {"decision_date": date, "mode": mode, "weights": {"A": 0.5}, **kw}


def test_log_is_append_only_hash_chained_and_tamper_evident(tmp_path):
    log = DecisionLog(tmp_path / "d.jsonl", min_date="2026-11-02")
    r1 = log.append(_rec("2026-11-02", "anon"))
    r2 = log.append(_rec("2026-11-02", "real"))
    assert r1["prev_hash"] == GENESIS and r2["prev_hash"] == r1["record_hash"] and log.verify() == 2
    with pytest.raises(DuplicateDecision):
        log.append(_rec("2026-11-02", "anon"))
    with pytest.raises(ValueError):
        log.append(_rec("2026-10-30", "anon"))                       # before start
    log.append(_rec("2026-11-09", "anon"))
    with pytest.raises(ValueError):
        log.append(_rec("2026-11-04", "anon"))                       # back in time within a mode
    lines = (tmp_path / "d.jsonl").read_text().splitlines()
    assert len(lines) == 3
    tampered = json.loads(lines[0])
    tampered["weights"] = {"A": 1.0}
    (tmp_path / "d.jsonl").write_text("\n".join([json.dumps(tampered)] + lines[1:]) + "\n")
    with pytest.raises(LogIntegrityError):
        log.verify()
    with pytest.raises(LogIntegrityError):                           # and appending to a broken log is refused
        log.append(_rec("2026-11-16", "anon"))


# ------------------------------------------------------------------ runner

def _setup(n=300, start_idx=250):
    m = rw_market(n_bars=n, tickers=TICKERS)
    cfg = ForwardConfig(start=str(m.dates[start_idx].date()), provider="mock", model="mock_ew")
    return m, cfg


def test_run_forward_decides_once_and_is_idempotent(tmp_path):
    m, cfg = _setup()
    now = lambda: decision_deadline(str(m.dates[250].date())) - dt.timedelta(hours=1)  # noqa: E731
    m1 = m.slice(end=m.dates[250])             # first decision bar = start bar
    new = run_forward(cfg, m1, directory=tmp_path, now=now)
    assert {r["mode"] for r in new} == {"anon", "real"} and all(r["decision_date"] == str(m.dates[250].date()) for r in new)
    f = tmp_path / "decisions.jsonl"
    snapshot = f.read_bytes()
    assert run_forward(cfg, m1, directory=tmp_path, now=now) == []
    assert f.read_bytes() == snapshot and len(f.read_text().splitlines()) == 2
    # next days: not a decision bar until a new ISO week; whatever is decided is never duplicated
    for t in range(251, 262):
        run_forward(cfg, m.slice(end=m.dates[t]), directory=tmp_path, now=now, enforce_deadline=False)
        run_forward(cfg, m.slice(end=m.dates[t]), directory=tmp_path, now=now, enforce_deadline=False)
    recs = DecisionLog(f).read()
    keys = [(r["decision_date"], r["mode"]) for r in recs]
    assert len(keys) == len(set(keys)) > 2 and DecisionLog(f).verify() == len(recs)
    assert all(r["decision_date"] >= cfg.start and r["model_served"] == "mock_ew" for r in recs)


def test_missed_day_is_a_hold_not_a_backfill(tmp_path):
    m, cfg = _setup()
    now = lambda: dt.datetime(2030, 1, 1, tzinfo=UTC)  # noqa: E731
    # run only on a later bar: the (missed) start-bar decision must not be made retroactively
    run_forward(cfg, m.slice(end=m.dates[256]), directory=tmp_path, now=now, enforce_deadline=False)
    dates = {r["decision_date"] for r in DecisionLog(tmp_path / "decisions.jsonl").read()}
    assert str(m.dates[250].date()) not in dates


def test_stale_decision_after_next_open_is_refused(tmp_path):
    m, cfg = _setup()
    m1 = m.slice(end=m.dates[250])
    late = lambda: decision_deadline(str(m.dates[250].date())) + dt.timedelta(minutes=1)  # noqa: E731
    with pytest.raises(StaleDecision):
        run_forward(cfg, m1, directory=tmp_path, now=late)
    assert not (tmp_path / "decisions.jsonl").exists()
    ok = lambda: decision_deadline(str(m.dates[250].date())) - dt.timedelta(minutes=1)  # noqa: E731
    assert len(run_forward(cfg, m1, directory=tmp_path, now=ok)) == 2


def test_decision_uses_only_data_up_to_the_bar(tmp_path):
    """Perturbing every bar after the decision bar cannot change the logged prompt/decision."""
    m, cfg = _setup()
    now = lambda: dt.datetime(2030, 1, 1, tzinfo=UTC)  # noqa: E731
    d1, d2 = tmp_path / "a", tmp_path / "b"
    a = run_forward(cfg, m.slice(end=m.dates[250]), directory=d1, now=now, enforce_deadline=False)
    pert = m.with_perturbation(m.dates[250], 1.7)
    b = run_forward(cfg, pert.slice(end=pert.dates[250]), directory=d2, now=now, enforce_deadline=False)
    assert [r["prompt_sha256"] for r in a] == [r["prompt_sha256"] for r in b]


def test_include_last_bar_matches_historical_engine_on_shared_bars():
    m = rw_market(n_bars=150, tickers=TICKERS)
    cfg = BacktestConfig(rebalance="weekly", warmup=100)
    base = run_backtest(EqualWeight(list(TICKERS), rebalance=True), m, cfg)
    live = run_backtest(EqualWeight(list(TICKERS), rebalance=True), m, cfg, include_last_bar=True)
    pd.testing.assert_series_equal(base.equity, live.equity)
    assert len(live.trades) == len(base.trades)                   # orders from the last bar stay pending, never filled


def test_bootstrap_gap_withholds_ci_when_n_small_and_detects_a_real_gap():
    small = pd.Series(np.random.default_rng(0).normal(0, 0.01, MIN_DAYS_FOR_CI - 1))
    assert bootstrap_gap(small)["ci"] is None
    big = pd.Series(np.random.default_rng(1).normal(0.002, 0.005, 250))
    g = bootstrap_gap(big)
    assert g["ci"] is not None and g["ci"][0] > 0
    zero = pd.Series(np.random.default_rng(2).normal(0, 0.01, 250))
    lo, hi = bootstrap_gap(zero)["ci"]
    assert lo < 0 < hi
