"""M4a LLM-agent tests. All offline: only the deterministic mock providers are ever called, and
every test uses a temporary cache directory so nothing touches data/llm_cache."""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
import pytest

from sab.data.universe import TICKERS
from sab.engine import BacktestConfig, MarketData, ZeroCost, load_market, run_backtest
from sab.engine.backtest import Context
from sab.engine.types import PortfolioState
from sab.llm import (AnonMapping, LLMAgent, LLMCache, assert_prompt_no_future, build_prompt, cache_key,
                     complete, consistency_check, parse_weights)
from sab.llm.parse import ParseError, decide_with_retry
from sab.llm.prompt import table_rows
from sab.llm.providers import approx_tokens, tickers_from_prompt
from sab.strategies import EqualWeight

TICKS = ("AAA", "BBB", "CCC", "DDD")


@pytest.fixture
def cache(tmp_path):
    return LLMCache(tmp_path / "llm_cache")


def rw_market(n_bars=400, tickers=TICKS, seed=0, start="2019-01-01") -> MarketData:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n_bars)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, size=(n_bars, len(tickers))), axis=0))
    open_ = close * (1 + rng.normal(0, 0.003, size=close.shape))
    return MarketData(pd.DataFrame(open_, index=dates, columns=list(tickers)),
                      pd.DataFrame(close, index=dates, columns=list(tickers)))


def ctx_at(m: MarketData, t: int, cash=1e6) -> Context:
    st = PortfolioState(cash=cash)
    st.mark(m.dates[t], m.close.iloc[t])
    return Context(m.close.iloc[: t + 1], m.open.iloc[: t + 1], st, t, ZeroCost)


class Recording:
    def __init__(self, inner):
        self.inner, self.name, self.decisions = inner, inner.name, {}

    @property
    def params(self):
        return self.inner.params

    def decisions_frame(self):
        return self.inner.decisions_frame()

    def on_bar(self, ctx):
        orders = self.inner.on_bar(ctx)
        self.decisions[ctx.bar_index] = orders
        return orders


# ----------------------------------------------------------------------------- import hygiene

def test_importing_sab_llm_needs_no_provider_sdk():
    assert not any(m.split(".")[0] in {"anthropic", "openai", "google"} for m in sys.modules), "provider SDK imported eagerly"


# ----------------------------------------------------------------------------- prompt / no-future

@pytest.mark.parametrize("date", ["2020-03-16", "2021-06-30", "2022-01-03", "2022-11-15", "2024-05-01"])
@pytest.mark.parametrize("mode", ["real", "anon"])
def test_prompt_contains_no_future_on_real_data(date, mode):
    m = load_market()
    t = int(m.dates.searchsorted(pd.Timestamp(date)))
    ctx = ctx_at(m, t)
    a = LLMAgent("mock", "mock", mode, cache=None)
    spec = a.prompt_for(ctx)
    assert spec.last_date == m.dates[t]
    info = assert_prompt_no_future(spec.text, m.dates[t], m.close, mode=mode, lookback=a.lookback)
    assert info["future_values"] > 1000               # the check actually had something to look for
    if mode == "real":
        assert str(m.dates[t].date()) in spec.text
        assert "AAPL" in spec.text
    else:
        assert "AAPL" not in spec.text and str(m.dates[t].date()) not in spec.text
        assert f"Day {t}" in spec.text


def test_no_future_assertion_has_teeth():
    m = load_market()
    t = int(m.dates.searchsorted(pd.Timestamp("2022-06-01")))
    spec = LLMAgent("mock", "mock", "real", cache=None).prompt_for(ctx_at(m, t))
    leaked = spec.text + f"\n{m.close['NVDA'].iloc[-1]:.2f}"   # 2026 NVDA close, far above anything in the 2022 window
    with pytest.raises(AssertionError):
        assert_prompt_no_future(leaked, m.dates[t], m.close, mode="real")
    with pytest.raises(AssertionError):
        assert_prompt_no_future(spec.text + f"\n{m.dates[t + 3].date()}", m.dates[t], m.close, mode="real")


def test_prompt_structure_and_output_spec():
    m = rw_market(300)
    spec = build_prompt(m.close.iloc[:200], {"weights": {"AAA": 0.3}, "cash": 7e5, "equity": 1e6, "initial_equity": 9e5},
                        "real", lookback=60)
    txt = spec.text
    assert txt.index("Tickers:") < txt.index("Recent closes") < txt.index("Trailing returns") < txt.index("Portfolio:") < txt.index('{"weights"')
    assert "AAA 30.0%" in txt and "Cash: 70.0%" in txt and "Equity vs start: +11.1%" in txt
    assert "sum must be <= 1" in txt and "<= 40 words" in txt
    assert tickers_from_prompt(txt) == list(TICKS)
    assert spec.n_rows == len(table_rows(60)) == 20              # 10 daily + 10 weekly-sampled rows
    assert table_rows(60)[-10:] == list(range(50, 60)) and table_rows(60)[:3] == [0, 5, 10]
    assert txt.count("\n" + str(m.dates[199].date())) == 1        # today's row present exactly once


def test_anon_mapping_bijective_reversible_and_seeded():
    mp = AnonMapping.make(list(TICKERS), seed=3)
    assert sorted(mp.to_anon.values()) == [f"Stock {i:02d}" for i in range(1, 16)]
    assert all(mp.real(mp.anon(t)) == t for t in TICKERS)
    assert AnonMapping.make(list(TICKERS), seed=3).to_anon == mp.to_anon
    assert AnonMapping.make(list(TICKERS), seed=4).to_anon != mp.to_anon
    assert AnonMapping.from_dict(mp.to_dict()).to_real == mp.to_real
    a = LLMAgent("mock", "mock", "anon", seed=3, cache=None)
    assert a.params["anon_mapping"]["ticker_to_anon"] == mp.to_anon


def test_anon_prompt_rebases_prices_and_hides_order():
    m = load_market()
    t = 400
    spec = LLMAgent("mock", "mock", "anon", cache=None).prompt_for(ctx_at(m, t))
    line = next(l for l in spec.text.splitlines() if l.startswith("Tickers:"))
    assert line == "Tickers: " + ", ".join(f"Stock {i:02d}" for i in range(1, 16))
    first_row = next(l for l in spec.text.splitlines() if l.startswith("Day "))
    assert set(first_row.split(" | ")[1:]) == {"100.00"}


# ----------------------------------------------------------------------------- parse

ALLOWED = ["AAPL", "MSFT", "NVDA"]


def test_parse_clean_json():
    r = parse_weights('{"weights": {"AAPL": 0.5, "MSFT": 0.3}, "rationale": "tech"}', ALLOWED)
    assert r.weights == {"AAPL": 0.5, "MSFT": 0.3, "NVDA": 0.0} and r.rationale == "tech"
    assert r.repairs["missing_zeroed"] == 1 and r.n_repairs == 1


def test_parse_fenced_and_prose():
    txt = 'Sure! Here you go:\n```json\n{"weights": {"AAPL": 0.4}, "rationale": "x"}\n```\nLet me know.'
    r = parse_weights(txt, ALLOWED)
    assert r.weights["AAPL"] == 0.4 and r.repairs["fence_stripped"] == 1


def test_parse_trailing_comma():
    r = parse_weights('{"weights": {"AAPL": 0.4, "NVDA": 0.2,}, "rationale": "y",}', ALLOWED)
    assert r.weights == {"AAPL": 0.4, "MSFT": 0.0, "NVDA": 0.2} and r.repairs["trailing_comma_fixed"] == 1


def test_parse_unknown_ticker_negative_and_sum_gt_one():
    r = parse_weights('{"weights": {"AAPL": 0.9, "TSLA": 0.5, "MSFT": -0.2, "nvda": 0.6}, "rationale": "z"}', ALLOWED)
    assert r.repairs["unknown_dropped"] == 1 and r.repairs["negative_clipped"] == 1 and r.repairs["renormalised"] == 1
    assert abs(sum(r.weights.values()) - 1.0) < 1e-12 and r.weights["MSFT"] == 0.0
    assert r.weights["AAPL"] == pytest.approx(0.6) and r.weights["NVDA"] == pytest.approx(0.4)   # case-folded key kept


def test_parse_percent_strings_and_missing_rationale():
    r = parse_weights('{"weights": {"AAPL": "40%", "MSFT": "abc"}}', ALLOWED)
    assert r.weights["AAPL"] == 1.0 and r.repairs["renormalised"] == 1          # "40%" -> 40 -> renormalised to 1.0
    assert r.repairs["non_numeric_dropped"] == 1 and r.repairs["rationale_missing"] == 1


@pytest.mark.parametrize("garbage", ["", "I cannot decide.", '{"foo": 1}', '{"weights": {"TSLA": 1}}', "{broken json"])
def test_parse_garbage_raises(garbage):
    with pytest.raises(ParseError):
        parse_weights(garbage, ALLOWED)


def test_retry_then_fallback_holds_previous_weights():
    calls = []

    class R:
        def __init__(self, text): self.text = text

    def ask_bad(p):
        calls.append(p)
        return R("nope")

    w, meta = decide_with_retry(ask_bad, "PROMPT", ALLOWED, prev_weights={"AAPL": 0.25})
    assert w == {"AAPL": 0.25, "MSFT": 0.0, "NVDA": 0.0}
    assert meta["attempts"] == 2 and meta["retried"] and meta["fallback"] and "could not be used" in calls[1]
    assert calls[1].startswith("PROMPT")

    def ask_second_ok(p):
        return R("nope") if "could not be used" not in p else R('{"weights": {"NVDA": 1}, "rationale": "ok"}')

    w, meta = decide_with_retry(ask_second_ok, "PROMPT", ALLOWED)
    assert w["NVDA"] == 1.0 and meta["retried"] and not meta["fallback"] and meta["attempts"] == 2


# ----------------------------------------------------------------------------- providers / cache

def test_mock_providers_are_deterministic_and_valid(cache):
    m = rw_market(300)
    p1 = build_prompt(m.close.iloc[:200], {"equity": 1e6, "cash": 1e6}, "real").text
    p2 = build_prompt(m.close.iloc[:201], {"equity": 1e6, "cash": 1e6}, "real").text
    for prov in ("mock", "mock_ew", "mock_random"):
        a = complete(prov, "x", p1, cache=cache, seed=0)
        b = complete(prov, "x", p1, cache=None, bypass_cache=True, seed=0)
        assert a.text == b.text                                                    # pure function of (prompt, seed)
        w = json.loads(a.text)["weights"]
        assert set(w) == set(TICKS) and all(v >= 0 for v in w.values()) and sum(w.values()) <= 1 + 1e-9
    assert complete("mock", "x", p1, cache=None).text != complete("mock", "x", p2, cache=None).text
    ew = json.loads(complete("mock_ew", "x", p1, cache=None).text)["weights"]
    assert all(v == pytest.approx(0.25) for v in ew.values())
    assert complete("mock", "mock_ew", p1, cache=None).text == complete("mock_ew", "mock_ew", p1, cache=None).text
    with pytest.raises(ValueError):
        complete("nope", "x", p1, cache=None)


def test_cache_hit_miss_and_key(cache):
    prompt = "Tickers: AAA, BBB\nwhatever"
    r1 = complete("mock", "m", prompt, cache=cache)
    assert not r1.cached and cache.misses == 1 and cache.writes == 1 and len(cache) == 1
    r2 = complete("mock", "m", prompt, cache=cache)
    assert r2.cached and r2.text == r1.text and cache.hits == 1
    fresh = LLMCache(cache.directory)                       # new process: read from disk
    assert fresh.get(cache_key("mock", "m", prompt)) is not None and fresh.hits == 1
    rec = json.loads(cache.path(cache_key("mock", "m", prompt)).read_text())
    assert rec["prompt"] == prompt and rec["response"]["text"] == r1.text and "timestamp" in rec
    # a different temperature / seed / model is a different key
    assert cache_key("mock", "m", prompt, 0.7) != cache_key("mock", "m", prompt, 0.0) != cache_key("mock", "m", prompt, 0.0, 1)
    r3 = complete("mock", "m", prompt, cache=cache, bypass_cache=True)
    assert not r3.cached and cache.hits == 1                # bypass skips the read but still writes


# ----------------------------------------------------------------------------- agent through the engine

def test_mock_ew_agent_matches_equal_weight_weekly(cache):
    m = load_market().slice("2021-06-01", "2022-12-31")
    a = LLMAgent("mock", "mock_ew", "anon", cache=cache)
    r_llm = run_backtest(a, m, BacktestConfig(rebalance="weekly", warmup=a.warmup))
    r_ew = run_backtest(EqualWeight(TICKERS, rebalance=True), m, BacktestConfig(rebalance="weekly", warmup=a.warmup))
    pd.testing.assert_series_equal(r_llm.equity, r_ew.equity, rtol=1e-9)
    assert len(r_llm.trades) == len(r_ew.trades)
    assert r_llm.decisions is not None and len(r_llm.decisions) == len(a.decisions)
    assert (r_llm.decisions["exposure"] - 1.0).abs().max() < 1e-9 and (r_llm.decisions["n_held"] == 15).all()


def test_replay_makes_zero_provider_calls_and_is_identical(cache):
    m = load_market().slice("2022-01-01", "2022-12-31")
    a1 = LLMAgent("mock", "mock", "anon", cache=cache)
    r1 = run_backtest(a1, m, BacktestConfig(rebalance="weekly", warmup=a1.warmup))
    assert a1.n_provider_calls == len(a1.decisions) > 10 and not a1.decisions["cache_hit"].any()
    a2 = LLMAgent("mock", "mock", "anon", cache=cache)
    r2 = run_backtest(a2, m, BacktestConfig(rebalance="weekly", warmup=a2.warmup))
    assert a2.n_provider_calls == 0 and a2.decisions["cache_hit"].all()
    pd.testing.assert_series_equal(r1.equity, r2.equity)
    assert list(a1.decisions["prompt_sha256"]) == list(a2.decisions["prompt_sha256"])
    assert r1.config["params"]["provider"] == "mock" and r1.config["params"]["anon_mapping"] is not None


@pytest.mark.parametrize("mode", ["real", "anon"])
def test_lookahead_perturbation_harness_llm_agent(mode, cache):
    """Identical to the M2/M3 harness: perturb every bar strictly after T*; equity, fills and
    decisions through T* must be identical, and later decisions must differ. With the hash-based
    mock this is a genuine proof that the prompt carries no future information: any future price
    in the prompt would change its sha256 and therefore the mock's weights before T*."""
    m = rw_market(300, seed=11)
    t_star = 200
    T = m.dates[t_star]
    outs = []
    for mk in (m, m.with_perturbation(T, 1.5, tickers=["AAA"])):
        s = Recording(LLMAgent("mock", "mock", mode, tickers=TICKS, cache=cache))
        r = run_backtest(s, mk, BacktestConfig(rebalance="daily", warmup=s.inner.warmup))
        outs.append((r, s.decisions, s.inner.decisions))
    (base, dec_b, log_b), (pert, dec_p, log_p) = outs
    pd.testing.assert_series_equal(base.equity.loc[:T], pert.equity.loc[:T])
    assert [f for f in base.trades if f.date <= T] == [f for f in pert.trades if f.date <= T]
    for t in range(min(dec_b), t_star + 1):
        assert dec_b[t] == dec_p[t], f"decision at bar {t} <= T* differs"
        assert log_b["prompt_sha256"].iloc[t - min(dec_b)] == log_p["prompt_sha256"].iloc[t - min(dec_b)]
    assert any(dec_b[t] != dec_p[t] for t in range(t_star + 1, max(dec_b) + 1)), "harness has no teeth"


def test_agent_orders_are_valid_and_invariants_hold(cache):
    m = rw_market(260, seed=5)
    s = Recording(LLMAgent("mock", "mock", "real", tickers=TICKS, cache=cache))
    r = run_backtest(s, m, BacktestConfig(rebalance="weekly", warmup=s.inner.warmup))
    assert s.decisions
    for orders in s.decisions.values():
        assert all(o.is_target and 0 <= o.target_weight <= 1 for o in orders) and sum(o.target_weight for o in orders) <= 1 + 1e-9
    assert (r.cash >= -1e-6).all() and (r.exposure <= 1 + 1e-6).all()
    assert s.inner.params["mode"] == "real" and s.inner.params["anon_mapping"] is None


def test_agent_falls_back_to_hold_on_garbage(cache, monkeypatch):
    import sab.llm.strategy as strat_mod
    from sab.llm.providers import LLMResponse
    m = rw_market(200, seed=2)
    a = LLMAgent("mock", "mock", "real", tickers=TICKS, cache=None)
    ctx = ctx_at(m, 150)
    a.decide(ctx)                                     # populates the log with a good decision
    monkeypatch.setattr(strat_mod, "complete", lambda *args, **kw: LLMResponse(text="I refuse.", provider="mock"))
    w, meta = a.decide(ctx)
    assert meta["fallback"] and meta["retried"] and meta["attempts"] == 2
    assert w == {t: 0.0 for t in TICKS}               # nothing held -> hold nothing
    row = a.decisions.iloc[-1]
    assert row["fallback"] and row["n_calls"] == 2 and row["error"]


def test_result_save_writes_decisions(cache, tmp_path):
    m = rw_market(200, seed=3)
    a = LLMAgent("mock", "mock", "anon", tickers=TICKS, cache=cache)
    r = run_backtest(a, m, BacktestConfig(rebalance="monthly", warmup=a.warmup))
    d = r.save(tmp_path, name="agent")
    dec = pd.read_parquet(d / "decisions.parquet")
    assert len(dec) == len(a.decisions) and json.loads(dec["weights"].iloc[0]) == a.decisions["weights"].iloc[0]
    cfg = json.loads((d / "config.json").read_text())
    assert cfg["params"]["anon_mapping"]["ticker_to_anon"]


# ----------------------------------------------------------------------------- consistency + tokens

def test_consistency_check_mock_vs_random(cache):
    m = rw_market(200, seed=4)
    ctx = ctx_at(m, 150)
    c = consistency_check(LLMAgent("mock", "mock", "anon", tickers=TICKS, cache=cache), ctx, n=3, temperature=0.7)
    assert c["fraction_identical"] == 1.0 and c["n_distinct"] == 1
    r = consistency_check(LLMAgent("mock_random", "x", "anon", tickers=TICKS, cache=cache), ctx, n=4, temperature=0.7)
    assert r["fraction_identical"] < 1.0 and r["n_distinct"] > 1
    assert len(cache) == 0                            # consistency checks never write the cache


def test_token_estimate_runs():
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "scripts"))
    import run_llm
    args = run_llm.argparse.Namespace(period=["2022"], mode="anon", lookback=60, rebalance="weekly", seed=0)
    run_llm.estimate(args)                            # prints; must not raise or call a provider
    txt = build_prompt(load_market().close.iloc[:300], {"equity": 1e6, "cash": 1e6}, "anon").text
    assert 800 < approx_tokens(txt) < 2500
