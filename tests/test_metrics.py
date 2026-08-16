"""M2 metrics tests on synthetic curves with known answers."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from sab.engine.types import Fill
from sab.metrics import metrics as M


def _series(values, start="2020-01-01") -> pd.Series:
    return pd.Series(list(values), index=pd.bdate_range(start, periods=len(values)), dtype=float)


def test_constant_growth_known_cagr_zero_drawdown():
    daily = 0.001
    eq = _series(100 * (1 + daily) ** np.arange(253))          # 252 bars of growth = 1 "year"
    assert M.cagr(eq) == pytest.approx((1 + daily) ** 252 - 1)
    mdd, dur = M.max_drawdown(eq)
    assert mdd == 0 and dur == 0
    assert M.volatility(eq) == pytest.approx(0.0, abs=1e-12)
    assert math.isnan(M.sharpe(eq))                             # zero vol -> undefined, not inf
    assert math.isnan(M.calmar(eq))                             # no drawdown -> undefined


def test_two_years_doubling_gives_41pct_cagr():
    eq = _series(np.linspace(100, 200, 505))                    # 504 bars = 2 years, +100%
    assert M.cagr(eq) == pytest.approx(math.sqrt(2) - 1, rel=1e-6)
    assert M.total_return(eq) == pytest.approx(1.0)


def test_known_drawdown_and_duration():
    # 100 -> 120 (peak) -> 90 (trough, -25%) -> 120 (recovered) -> 130
    eq = _series([100, 110, 120, 105, 90, 100, 115, 120, 130])
    mdd, dur = M.max_drawdown(eq)
    assert mdd == pytest.approx(-0.25)
    assert dur == 4                                             # bars 105, 90, 100, 115 are below peak
    assert M.calmar(eq) == pytest.approx(M.cagr(eq) / 0.25)


def test_unrecovered_drawdown_duration_runs_to_end():
    eq = _series([100, 120, 100, 90, 95])
    mdd, dur = M.max_drawdown(eq)
    assert mdd == pytest.approx(-0.25) and dur == 3


def test_sharpe_and_sortino_hand_computed():
    rets = np.array([0.01, -0.005, 0.02, -0.01, 0.005])
    eq = _series(100 * np.cumprod(np.r_[1.0, 1 + rets]))
    r = pd.Series(rets)
    assert M.sharpe(eq) == pytest.approx(r.mean() / r.std(ddof=1) * math.sqrt(252))
    down = math.sqrt(np.mean(np.minimum(rets, 0) ** 2))
    assert M.sortino(eq) == pytest.approx(r.mean() / down * math.sqrt(252))
    # rf lowers Sharpe: with rf = 5% the per-bar excess return drops by 0.05/252
    assert M.sharpe(eq, rf=0.05) < M.sharpe(eq, rf=0.0)
    assert M.sharpe(eq, rf=0.05) == pytest.approx((r - 0.05 / 252).mean() / r.std(ddof=1) * math.sqrt(252))


def test_win_rate_and_costs_from_trades():
    d = pd.Timestamp("2020-01-02")
    trades = [
        Fill(d, "A", "BUY", 10, 100.0, 1.0, 0.5),
        Fill(d, "A", "SELL", 5, 110.0, 1.0, 0.5),      # win
        Fill(d, "A", "SELL", 5, 90.0, 1.0, 0.5),       # loss
        Fill(d, "B", "BUY", 1, 50.0, 0.1, 0.0),
        Fill(d, "B", "SELL", 1, 60.0, 0.1, 0.0),       # win
    ]
    assert M.win_rate(trades) == pytest.approx(2 / 3)
    assert M.total_costs(trades) == pytest.approx(4.7)
    assert math.isnan(M.win_rate([]))


def test_annualised_turnover():
    tv = _series(np.zeros(252))
    tv.iloc[10] = 0.5                                           # one half-portfolio round trip a year
    assert M.annualised_turnover(tv) == pytest.approx(0.5)


def test_summary_table_shape_and_formatting():
    from sab.engine import BacktestConfig, ZeroCost, run_backtest
    from sab.strategies.buy_and_hold import EqualWeight
    from tests.test_engine import make_market

    m = make_market(60)
    res = {
        "with": run_backtest(EqualWeight(m.tickers), m, BacktestConfig()),
        "without": run_backtest(EqualWeight(m.tickers), m, BacktestConfig(cost_model=ZeroCost)),
    }
    df = M.summary_table(res, rf=0.0)
    assert list(df.index) == ["with", "without"]
    for col in ("CAGR", "Sharpe", "Sortino", "MaxDD", "DD_days", "Calmar", "Turnover", "Exposure", "WinRate", "Costs"):
        assert col in df.columns
    assert df.loc["with", "Costs"] > 0 and df.loc["without", "Costs"] == 0
    assert df.loc["with", "CAGR"] < df.loc["without", "CAGR"]
    assert "rf=0" in df.index.name
    md = M.format_table(df)
    assert md.startswith("|") and "with" in md
