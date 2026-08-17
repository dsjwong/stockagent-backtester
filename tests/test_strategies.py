"""M3 strategy tests: valid orders, lookahead-perturbation harness over every classical
strategy, and behavioural checks on synthetic series (trend fully invested on a monotone
up-series / cash on a monotone down-series; mean reversion buys the shocked ticker)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sab.engine import BacktestConfig, MarketData, Order, ZeroCost, load_market, run_backtest
from sab.strategies import CLASSICAL, MeanReversion, MomentumTopK, TrendFollowing, WeightedTrend
from sab.strategies.weighted_trend import cap_weights

TICKS = ("AAA", "BBB", "CCC", "DDD")


def rw_market(n_bars: int = 400, tickers=TICKS, seed: int = 0, start="2019-01-01") -> MarketData:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n_bars)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, size=(n_bars, len(tickers))), axis=0))
    open_ = close * (1 + rng.normal(0, 0.003, size=close.shape))
    return MarketData(pd.DataFrame(open_, index=dates, columns=list(tickers)),
                      pd.DataFrame(close, index=dates, columns=list(tickers)))


def path_market(paths: dict[str, np.ndarray], start="2019-01-01") -> MarketData:
    n = len(next(iter(paths.values())))
    dates = pd.bdate_range(start, periods=n)
    close = pd.DataFrame(paths, index=dates)
    return MarketData(close * 1.001, close)


def make(cls, tickers, **kw):
    """Small lookbacks so synthetic markets stay short; same code paths as the defaults."""
    if cls is MeanReversion:
        return MeanReversion(tickers, lookback=20, **kw)
    if cls is TrendFollowing:
        return TrendFollowing(tickers, fast=10, slow=40, **kw)
    if cls is WeightedTrend:
        return WeightedTrend(tickers, horizons=(5, 10, 20, 40), **kw)
    return MomentumTopK(tickers, k=2, lookback=40, skip=5, **kw)


class Recording:
    """Wraps a strategy and records every order list, per bar."""
    def __init__(self, inner):
        self.inner, self.name, self.decisions = inner, inner.name, {}

    def on_bar(self, ctx):
        orders = self.inner.on_bar(ctx)
        self.decisions[ctx.bar_index] = orders
        return orders


# ----------------------------------------------------------------------------- validity

@pytest.mark.parametrize("cls", CLASSICAL)
@pytest.mark.parametrize("rebalance", ["daily", "weekly", "monthly"])
def test_orders_are_valid_targets_and_engine_invariants_hold(cls, rebalance):
    m = rw_market(300, seed=7)
    strat = Recording(make(cls, TICKS))
    res = run_backtest(strat, m, BacktestConfig(rebalance=rebalance, warmup=strat.inner.warmup))
    assert strat.decisions, "strategy never decided"
    n_nonempty = 0
    for bar, orders in strat.decisions.items():
        assert all(isinstance(o, Order) and o.is_target for o in orders)
        ws = [o.target_weight for o in orders]
        assert all(0.0 <= w <= 1.0 for w in ws)
        assert sum(ws) <= 1.0 + 1e-9
        assert len({o.ticker for o in orders}) == len(orders)          # one order per ticker
        n_nonempty += bool(orders)
    assert n_nonempty > 0
    assert (res.cash >= -1e-6).all()
    assert (res.exposure <= 1.0 + 1e-6).all()
    assert (res.weights >= -1e-12).all().all()


@pytest.mark.parametrize("cls", CLASSICAL)
def test_params_and_warmup_surface_in_result_config(cls):
    m = rw_market(200, seed=1)
    strat = make(cls, TICKS)
    res = run_backtest(strat, m, BacktestConfig(rebalance=strat.rebalance, warmup=strat.warmup))
    assert res.config["params"] == strat.params and res.config["warmup"] == strat.warmup
    assert strat.rebalance in ("daily", "weekly", "monthly")


def test_defaults_are_textbook_and_warmup_is_max_lookback():
    assert (MeanReversion().lookback, MeanReversion().z_entry, MeanReversion().warmup) == (20, -1.0, 20)
    assert (TrendFollowing().fast, TrendFollowing().slow, TrendFollowing().warmup) == (50, 200, 200)
    assert WeightedTrend().horizons == (21, 63, 126, 252) and WeightedTrend().warmup == 252
    assert (MomentumTopK().k, MomentumTopK().lookback, MomentumTopK().skip, MomentumTopK().warmup) == (5, 252, 21, 252)


def test_strategies_ignore_non_universe_columns():
    """SPY sits in the market frame during the reports but must never be traded."""
    m = rw_market(120, tickers=TICKS + ("SPY",), seed=3)
    for cls in CLASSICAL:
        strat = Recording(make(cls, TICKS))
        run_backtest(strat, m, BacktestConfig(rebalance="daily", warmup=strat.inner.warmup))
        assert all(o.ticker != "SPY" for orders in strat.decisions.values() for o in orders)


# ----------------------------------------------------------------------------- lookahead

@pytest.mark.parametrize("cls", CLASSICAL)
def test_lookahead_perturbation_harness(cls):
    """Same harness as M2 (test_engine): perturb every bar strictly after T*; equity marked
    through T*, fills dated <= T* and decisions through T* must be identical; decisions after
    T* must eventually differ (proves the strategy actually reads the perturbed data)."""
    m = rw_market(260, seed=11)
    t_star = 150
    T = m.dates[t_star]
    outs = []
    for mk in (m, m.with_perturbation(T, 1.5, tickers=["AAA"])):
        s = Recording(make(cls, TICKS))
        r = run_backtest(s, mk, BacktestConfig(rebalance="daily", warmup=s.inner.warmup))
        outs.append((r, s.decisions))
    (base, dec_b), (pert, dec_p) = outs
    pd.testing.assert_series_equal(base.equity.loc[:T], pert.equity.loc[:T])
    assert [f for f in base.trades if f.date <= T] == [f for f in pert.trades if f.date <= T]
    for t in range(min(dec_b), t_star + 1):
        assert dec_b[t] == dec_p[t], f"{cls.__name__}: decision at bar {t} <= T* differs"
    assert any(dec_b[t] != dec_p[t] for t in range(t_star + 1, max(dec_b) + 1)), "harness has no teeth"


# ----------------------------------------------------------------------------- behaviour

def test_trend_fully_invested_on_monotone_up_and_cash_on_monotone_down():
    n = 120
    up = 100 * 1.003 ** np.arange(n)
    down = 100 * 0.997 ** np.arange(n)
    for sizing in ("equal", "slots"):
        m_up = path_market({"AAA": up, "BBB": up * 1.1, "CCC": up * 0.9})
        s = TrendFollowing(("AAA", "BBB", "CCC"), fast=10, slow=40, sizing=sizing)
        r = run_backtest(s, m_up, BacktestConfig(rebalance="daily", warmup=s.warmup, cost_model=ZeroCost))
        after = r.exposure.iloc[s.warmup + 2:]           # first fill lands at open[warmup+1]
        assert (after > 0.99).all(), f"{sizing}: not fully invested on an up-trend: min {after.min():.4f}"

        m_dn = path_market({"AAA": down, "BBB": down * 1.1, "CCC": down * 0.9})
        s = TrendFollowing(("AAA", "BBB", "CCC"), fast=10, slow=40, sizing=sizing)
        r = run_backtest(s, m_dn, BacktestConfig(rebalance="daily", warmup=s.warmup, cost_model=ZeroCost))
        assert (r.exposure == 0).all() and not r.trades, f"{sizing}: traded on a down-trend"


def test_trend_slots_partial_exposure_when_some_names_trend():
    n = 120
    up, down = 100 * 1.003 ** np.arange(n), 100 * 0.997 ** np.arange(n)
    m = path_market({"AAA": up, "BBB": down, "CCC": up, "DDD": down})
    s = TrendFollowing(TICKS, fast=10, slow=40, sizing="slots")
    r = run_backtest(s, m, BacktestConfig(rebalance="daily", warmup=s.warmup, cost_model=ZeroCost))
    tail = r.weights.iloc[s.warmup + 2:]
    assert (tail[["BBB", "DDD"]] == 0).all().all()
    assert tail[["AAA", "CCC"]].gt(0.24).all().all() and (r.exposure.iloc[s.warmup + 2:] < 0.51).all()


def test_mean_reversion_buys_the_shocked_ticker():
    """Flat market with a little noise; AAA drops 8% on bar 60. At the next decision AAA's
    z-score is far below -1 and nothing else is, so the strategy goes 100% AAA."""
    rng = np.random.default_rng(5)
    n = 80
    base = 100 * (1 + rng.normal(0, 0.002, size=(n, 3))).cumprod(axis=0)
    base[60:, 0] *= 0.92
    m = path_market({"AAA": base[:, 0], "BBB": base[:, 1], "CCC": base[:, 2]})
    s = Recording(MeanReversion(("AAA", "BBB", "CCC"), lookback=20, z_entry=-1.0))
    run_backtest(s, m, BacktestConfig(rebalance="daily", warmup=20, cost_model=ZeroCost))
    orders = s.decisions[60]
    assert [(o.ticker, o.target_weight) for o in orders if o.target_weight > 0] == [("AAA", 1.0)]


def test_mean_reversion_all_cash_when_nothing_is_stretched():
    up = 100 * 1.001 ** np.arange(60)                    # steady rise: price always > SMA
    m = path_market({"AAA": up, "BBB": up})
    s = MeanReversion(("AAA", "BBB"), lookback=20)
    r = run_backtest(s, m, BacktestConfig(rebalance="daily", warmup=20, cost_model=ZeroCost))
    assert not r.trades and (r.exposure == 0).all()


def test_weighted_trend_weights_follow_momentum_and_cap():
    n = 60
    strong = 100 * 1.004 ** np.arange(n)
    weak = 100 * 1.001 ** np.arange(n)
    down = 100 * 0.998 ** np.arange(n)
    m = path_market({"AAA": strong, "BBB": weak, "CCC": down})
    s = Recording(WeightedTrend(("AAA", "BBB", "CCC"), horizons=(5, 10, 20, 40), max_weight=None))
    run_backtest(s, m, BacktestConfig(rebalance="daily", warmup=40, cost_model=ZeroCost))
    w = {o.ticker: o.target_weight for o in s.decisions[45]}
    assert w["AAA"] > w["BBB"] > 0 and "CCC" not in w and abs(sum(w.values()) - 1) < 1e-9
    capped = cap_weights(pd.Series({"a": 0.7, "b": 0.2, "c": 0.1}), 0.4)
    assert capped.max() <= 0.4 + 1e-12 and abs(capped.sum() - 1) < 1e-9 and capped["b"] > capped["c"]


def test_momentum_top_k_picks_the_k_best_12_1():
    n = 60
    paths = {t: 100 * (1 + g) ** np.arange(n) for t, g in zip(TICKS, (0.004, 0.003, 0.0, -0.002))}
    m = path_market(paths)
    s = Recording(MomentumTopK(TICKS, k=2, lookback=40, skip=5))
    run_backtest(s, m, BacktestConfig(rebalance="daily", warmup=40, cost_model=ZeroCost))
    assert sorted((o.ticker, o.target_weight) for o in s.decisions[45] if o.target_weight > 0) == [("AAA", 0.5), ("BBB", 0.5)]


def test_exit_orders_flatten_positions():
    """When a name drops out of the selection the strategy emits an explicit 0 target and the
    engine sells it (no orphaned positions)."""
    n = 120
    up_then_down = np.r_[100 * 1.005 ** np.arange(60), 100 * 1.005 ** 59 * 0.99 ** np.arange(1, 61)]
    flat_up = 100 * 1.001 ** np.arange(n)
    m = path_market({"AAA": up_then_down, "BBB": flat_up})
    s = TrendFollowing(("AAA", "BBB"), fast=5, slow=20)
    r = run_backtest(s, m, BacktestConfig(rebalance="daily", warmup=20, cost_model=ZeroCost))
    assert r.weights["AAA"].iloc[40] > 0.4 and r.weights["AAA"].iloc[-1] == 0.0
    assert any(f.ticker == "AAA" and f.side == "SELL" for f in r.trades)


@pytest.mark.parametrize("cls", CLASSICAL)
def test_real_data_smoke_and_determinism(cls):
    m = load_market().slice("2022-01-01", "2023-12-31")
    a = run_backtest(cls(), m, BacktestConfig(rebalance="weekly", warmup=cls().warmup))
    b = run_backtest(cls(), m, BacktestConfig(rebalance="weekly", warmup=cls().warmup))
    pd.testing.assert_series_equal(a.equity, b.equity)
    assert (a.cash >= -1e-6).all() and (a.exposure <= 1 + 1e-6).all()
