"""M2 engine tests: fill timing, lookahead impossibility, costs, cash/weights invariants,
determinism. Synthetic markets are used so expected numbers are exact; one test runs on the
real snapshot to make sure the invariants hold there too.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sab.engine import (
    BacktestConfig, Context, CostModel, MarketData, Order, PortfolioState, ZeroCost,
    load_market, run_backtest,
)
from sab.engine.broker import Broker
from sab.engine.types import Fill
from sab.strategies.buy_and_hold import EqualWeight

# ----------------------------------------------------------------------------- helpers

def make_market(n_bars: int = 60, tickers=("AAA", "BBB", "CCC"), seed: int = 0,
                start="2020-01-01") -> MarketData:
    """Synthetic random-walk market with open != close so open/close mixups are visible."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n_bars)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, size=(n_bars, len(tickers))), axis=0))
    open_ = close * (1 + rng.normal(0, 0.003, size=close.shape))
    return MarketData(pd.DataFrame(open_, index=dates, columns=list(tickers)),
                      pd.DataFrame(close, index=dates, columns=list(tickers)))


def flat_market(n_bars: int = 10, price: float = 100.0) -> MarketData:
    dates = pd.bdate_range("2020-01-01", periods=n_bars)
    df = pd.DataFrame(price, index=dates, columns=["AAA"])
    return MarketData(df.copy(), df.copy())


class ScriptedStrategy:
    """Emits a fixed list of orders on a given bar index, nothing otherwise."""
    name = "scripted"

    def __init__(self, script: dict[int, list[Order]]) -> None:
        self.script = script

    def on_bar(self, ctx: Context) -> list[Order]:
        return list(self.script.get(ctx.bar_index, []))


class MomentumEveryBar:
    """Data-dependent daily strategy: weight ~ rank of trailing 5-bar return. Used to make
    the lookahead tests bite (perturbing future prices WOULD change later decisions).
    Records every decision so tests can compare decisions, not just fills."""
    name = "momo"

    def __init__(self) -> None:
        self.decisions: dict[int, list[Order]] = {}

    def on_bar(self, ctx: Context) -> list[Order]:
        px = ctx.prices(6)
        if len(px) < 6:
            return []
        ret = px.iloc[-1] / px.iloc[0] - 1
        ranks = ret.rank()                      # 1..N
        w = ranks / ranks.sum()                 # sums to 1
        orders = [Order.target(t, float(w[t])) for t in ctx.tickers]
        self.decisions[ctx.bar_index] = orders
        return orders


class Peeker:
    """Tries to see the future; records what it could see."""
    name = "peeker"

    def __init__(self) -> None:
        self.seen: list[tuple[int, int, pd.Timestamp]] = []

    def on_bar(self, ctx: Context) -> list[Order]:
        n_all = len(ctx.prices())
        n_big = len(ctx.prices(10**9))
        assert n_all == n_big == len(ctx) == ctx.bar_index + 1
        assert ctx.prices().index[-1] == ctx.date
        assert ctx.opens().index[-1] == ctx.date
        assert not hasattr(ctx, "market") and not hasattr(ctx, "_market")
        self.seen.append((ctx.bar_index, n_all, ctx.prices().index[-1]))
        return []


# ----------------------------------------------------------------------------- fill timing

def test_order_fills_at_next_open_never_same_close():
    m = make_market(20)
    strat = ScriptedStrategy({3: [Order.market("AAA", "BUY", 10)]})
    res = run_backtest(strat, m, BacktestConfig(rebalance="daily", cost_model=ZeroCost))
    assert len(res.trades) == 1
    f = res.trades[0]
    assert f.date == m.dates[4]                       # bar t+1
    assert f.price == pytest.approx(m.open.iloc[4]["AAA"])
    assert f.price != pytest.approx(m.close.iloc[3]["AAA"])
    assert f.price != pytest.approx(m.open.iloc[3]["AAA"])
    # equity at close[3] is untouched (order not yet filled), position shows from bar 4
    assert res.weights.iloc[3]["AAA"] == 0
    assert res.weights.iloc[4]["AAA"] > 0
    assert res.equity.iloc[3] == pytest.approx(res.config["initial_cash"])


def test_last_bar_never_emits_orders():
    m = make_market(10)
    strat = ScriptedStrategy({9: [Order.market("AAA", "BUY", 1)]})
    res = run_backtest(strat, m, BacktestConfig(rebalance="daily", cost_model=ZeroCost))
    assert res.trades == []


# ----------------------------------------------------------------------------- lookahead

def _run(m: MarketData, cost: CostModel = CostModel()):
    strat = MomentumEveryBar()
    res = run_backtest(strat, m, BacktestConfig(rebalance="daily", warmup=5, cost_model=cost))
    res.decisions = strat.decisions            # type: ignore[attr-defined]  (test-only attach)
    return res


def _fills_up_to(trades: list[Fill], date) -> list[Fill]:
    return [f for f in trades if f.date <= date]


def test_lookahead_impossible_perturb_bars_after_T_star():
    """Perturb EVERY bar strictly after T* (open and close of AAA x1.5; one ticker so a
    rank-based strategy actually changes its mind afterwards).

    Exactly what must be identical: equity marked at closes <= T*, and every fill dated
    <= T* (those were decided at <= T*-1 and filled at opens <= T*). The fills at T*+1's
    OPEN are decided at T* from identical information, but they execute at a PERTURBED
    open, so their price (and integer qty, sized off open) may differ -- that is the first
    bar that is allowed to differ. See the sibling test for the T*+1-inclusive claim.
    """
    m = make_market(80, seed=1)
    t_star = 40
    T_star = m.dates[t_star]
    base, pert = _run(m), _run(m.with_perturbation(T_star, 1.5, tickers=["AAA"]))

    pd.testing.assert_series_equal(base.equity.loc[:T_star], pert.equity.loc[:T_star])
    assert _fills_up_to(base.trades, T_star) == _fills_up_to(pert.trades, T_star)
    # ... and the test has teeth: after T* things DO diverge
    assert not np.allclose(base.equity.loc[m.dates[t_star + 2]:], pert.equity.loc[m.dates[t_star + 2]:])
    later_base = [f for f in base.trades if f.date > m.dates[t_star + 1]]
    later_pert = [f for f in pert.trades if f.date > m.dates[t_star + 1]]
    assert later_base != later_pert


def test_lookahead_impossible_perturb_bars_from_T_star_plus_2():
    """Perturb bars >= T*+2 only. Then decisions at T* AND their fills at open[T*+1] AND
    the mark at close[T*+1] are all on unperturbed data => identical through T*+1
    inclusive. Fills at open[T*+2] (decided at T*+1 from identical data) execute at a
    perturbed price and are the first thing that may differ."""
    m = make_market(80, seed=2)
    t_star = 40
    T1 = m.dates[t_star + 1]
    base, pert = _run(m), _run(m.with_perturbation(T1, 1.5, tickers=["AAA"]))     # perturbs > T*+1

    pd.testing.assert_series_equal(base.equity.loc[:T1], pert.equity.loc[:T1])
    assert _fills_up_to(base.trades, T1) == _fills_up_to(pert.trades, T1)
    # the DECISIONS (target weights) through T*+1 inclusive are identical ...
    for t in range(5, t_star + 2):
        assert base.decisions[t] == pert.decisions[t], f"decision at bar {t} differs"
    # ... fills at open[T*+2] execute at x1.5 prices (qty/side may then differ: sized off open) ...
    d2 = m.dates[t_star + 2]
    fills_b = {(f.ticker, f.side): f.price for f in base.trades if f.date == d2}
    fills_p = {(f.ticker, f.side): f.price for f in pert.trades if f.date == d2}
    common = set(fills_b) & set(fills_p)
    assert common, "expected some same-side fills at T*+2 to compare"
    for key in common:
        assert fills_p[key] == pytest.approx(fills_b[key] * (1.5 if key[0] == "AAA" else 1.0))
    # ... and later decisions do change (the test has teeth)
    assert any(base.decisions[t] != pert.decisions[t] for t in range(t_star + 2, 79))


def test_context_never_exposes_future_rows():
    m = make_market(30)
    strat = Peeker()
    run_backtest(strat, m, BacktestConfig(rebalance="daily", warmup=2))
    assert [s[0] for s in strat.seen] == list(range(2, 29))          # bars 2..T-2
    for bar, n, last_date in strat.seen:
        assert n == bar + 1 and last_date == m.dates[bar]


def test_context_slice_is_independent_of_full_frame_length():
    """Same strategy sees the same context whether or not the market has more future bars."""
    long = make_market(50, seed=3)
    short = MarketData(long.open.iloc[:30], long.close.iloc[:30])
    p_long, p_short = Peeker(), Peeker()
    run_backtest(p_long, long, BacktestConfig(rebalance="daily"))
    run_backtest(p_short, short, BacktestConfig(rebalance="daily"))
    assert p_long.seen[:len(p_short.seen)] == p_short.seen


# ----------------------------------------------------------------------------- costs

def test_round_trip_cost_is_exact():
    """Buy 1 share at open 100 then sell it at open 100 with 10+5 bps per side.
    buy:  price 100.05, commission 0.10005 -> cash -100.15005
    sell: price  99.95, commission 0.09995 -> cash  +99.85005
    round trip = 0.30 = 30 bps of $100 (2 x (10 + 5) bps)."""
    m = flat_market(10, price=100.0)
    strat = ScriptedStrategy({1: [Order.market("AAA", "BUY", 1)], 3: [Order.market("AAA", "SELL", 1)]})
    res = run_backtest(strat, m, BacktestConfig(rebalance="daily", initial_cash=1000.0))
    buy, sell = res.trades
    assert (buy.price, buy.commission, buy.slippage) == pytest.approx((100.05, 0.10005, 0.05))
    assert (sell.price, sell.commission, sell.slippage) == pytest.approx((99.95, 0.09995, 0.05))
    assert res.equity.iloc[-1] == pytest.approx(1000.0 - 0.30)
    assert sum(f.total_cost for f in res.trades) == pytest.approx(0.30)


def test_zero_cost_round_trip_is_free():
    m = flat_market(10)
    strat = ScriptedStrategy({1: [Order.market("AAA", "BUY", 3)], 3: [Order.market("AAA", "SELL", 3)]})
    res = run_backtest(strat, m, BacktestConfig(rebalance="daily", initial_cash=1000.0, cost_model=ZeroCost))
    assert res.equity.iloc[-1] == pytest.approx(1000.0)


def test_with_costs_underperforms_without_by_exactly_costs_paid_when_prices_flat():
    m = flat_market(20)
    strat = ScriptedStrategy({i: [Order.market("AAA", "BUY" if i % 2 else "SELL", 2)] for i in range(1, 15)})
    with_c = run_backtest(strat, m, BacktestConfig(rebalance="daily", initial_cash=1000.0))
    assert with_c.equity.iloc[-1] == pytest.approx(1000.0 - sum(f.total_cost for f in with_c.trades))


# ----------------------------------------------------------------------------- invariants

def _check_invariants(res):
    assert (res.cash >= -1e-9).all(), "cash went negative"
    assert (res.exposure <= 1 + 1e-9).all(), "weights sum > 1"
    assert (res.weights >= 0).all().all(), "short position appeared"
    assert (res.equity > 0).all()


def test_cash_never_negative_and_weights_sum_le_one_synthetic():
    m = make_market(120, tickers=tuple(f"T{i}" for i in range(8)), seed=4)
    for cost in (CostModel(), ZeroCost):
        _check_invariants(_run(m, cost))
        _check_invariants(run_backtest(EqualWeight(m.tickers), m, BacktestConfig(cost_model=cost)))


def test_full_investment_fits_in_cash_after_costs():
    """Target weights summing to exactly 1 must not need scaling on the first buy:
    sizing uses the all-in buy price (open + slippage + commission)."""
    m = make_market(5, seed=5)
    res = run_backtest(EqualWeight(m.tickers), m, BacktestConfig())
    assert res.config["n_scaled_down"] == 0
    assert res.cash.iloc[-1] >= 0
    assert 0.98 < res.exposure.iloc[-1] <= 1


def test_broker_scales_down_instead_of_borrowing():
    m = flat_market(5, price=100.0)
    state = PortfolioState(cash=250.0)
    state.mark(m.dates[0], m.close.iloc[0])
    broker = Broker(ZeroCost)
    fills = broker.execute([Order.market("AAA", "BUY", 10)], state, m.open.iloc[1], m.dates[1])
    assert fills[0].qty == 2 and state.cash == pytest.approx(50.0)
    assert broker.n_scaled_down == 1 and broker.events


def test_broker_rejects_shorts_and_leverage():
    with pytest.raises(ValueError):
        Order.target("AAA", -0.1)
    with pytest.raises(ValueError):
        Order.target("AAA", 1.5)
    m = flat_market(3)
    state = PortfolioState(cash=1000.0)
    state.mark(m.dates[0], m.close.iloc[0])
    b = Broker(ZeroCost)
    with pytest.raises(ValueError, match="sum"):
        m2 = make_market(3)
        s2 = PortfolioState(cash=1000.0); s2.mark(m2.dates[0], m2.close.iloc[0])
        b.execute([Order.target("AAA", 0.6), Order.target("BBB", 0.6)], s2, m2.open.iloc[1], m2.dates[1])
    # selling more than held is capped, never goes short
    fills = b.execute([Order.market("AAA", "SELL", 5)], state, m.open.iloc[1], m.dates[1])
    assert fills == [] and b.n_sell_capped == 1


def test_real_data_equal_weight_invariants():
    m = load_market(start="2021-01-01", end="2021-12-31")
    res = run_backtest(EqualWeight(), m, BacktestConfig())
    _check_invariants(res)
    assert res.trades[0].date == m.dates[1]           # first fill at second bar's open
    assert res.equity.iloc[0] == res.config["initial_cash"]


# ----------------------------------------------------------------------------- determinism

def test_same_run_twice_is_identical():
    m = load_market(start="2022-01-01", end="2022-12-31")
    a = run_backtest(EqualWeight(), m, BacktestConfig())
    b = run_backtest(EqualWeight(), m, BacktestConfig())
    pd.testing.assert_series_equal(a.equity, b.equity)
    pd.testing.assert_frame_equal(a.weights, b.weights)
    assert a.trades == b.trades
    assert a.to_dict() == b.to_dict()


def test_result_roundtrip_save(tmp_path):
    m = make_market(30)
    res = run_backtest(EqualWeight(m.tickers), m, BacktestConfig(rebalance="weekly"))
    d = res.save(tmp_path)
    eq = pd.read_parquet(d / "equity.parquet")["equity"]
    pd.testing.assert_series_equal(eq, res.equity, check_names=False, check_freq=False)
    assert len(pd.read_parquet(d / "trades.parquet")) == len(res.trades)


def test_rebalance_frequency_controls_decision_count():
    from sab.engine.backtest import rebalance_bars
    dates = pd.bdate_range("2021-01-01", periods=250)
    daily = rebalance_bars(dates, "daily", 0)
    weekly = rebalance_bars(dates, "weekly", 0)
    monthly = rebalance_bars(dates, "monthly", 0)
    assert len(daily) == 249 and 50 <= len(weekly) <= 53 and len(monthly) == 12
    assert 0 in monthly and 249 not in daily
