"""M3: classical strategies vs equal-weight benchmarks, four periods, with and without costs,
plus a cost-sensitivity sweep. Offline (committed snapshots only). Run:

    .venv/bin/python scripts/run_classical.py

Writes reports/M3-results.md, reports/fig/equity_classical_full.png, equity_2021.png,
equity_2022.png, equity_2023_2026.png and one folder per run under reports/runs/.

Period handling: every strategy in a period is *entered from cash on the period's first bar*
(decisions.md #17) but is handed `warmup` bars of pre-period history so its signal is live on
day one (a 200-day SMA strategy started blind in January would sit in cash until October).
The market is sliced to [first_bar - warmup, end], `BacktestConfig.warmup` points at the
period's first bar, and the result is trimmed back to the period. Because WeightedTrend and
MomTop5 need 252 bars, "full" is 2019-01-02 .. 2026-08-14 (the M2 report keeps 2018 for the
benchmarks alone).
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib
import matplotlib.ticker
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sab.data.universe import BENCHMARK, TICKERS  # noqa: E402
from sab.engine import BacktestConfig, BacktestResult, CostModel, MarketData, ZeroCost, load_market, run_backtest  # noqa: E402
from sab.metrics.metrics import cagr, compute_metrics, format_table, summary_table  # noqa: E402
from sab.strategies import EqualWeight, MeanReversion, MomentumTopK, SpyReference, TrendFollowing, WeightedTrend  # noqa: E402

REPORTS = ROOT / "reports"
FIG = REPORTS / "fig"
RUNS = REPORTS / "runs"

PERIODS: dict[str, tuple[str | None, str | None]] = {
    "full (2019-01 .. 2026-08)": ("2019-01-01", None),
    "2021 (bull)": ("2021-01-01", "2021-12-31"),
    "2022 (drawdown)": ("2022-01-01", "2022-12-31"),
    "2023-01 .. 2026-08 (recovery)": ("2023-01-01", None),
}
COSTS: dict[str, CostModel] = {"with costs (10+5 bps)": CostModel(10.0, 5.0), "no costs": ZeroCost}
# round-trip bps -> per-side (commission, slippage); commission:slippage kept at 2:1 like the default
SENSITIVITY_RT_BPS = {0: (0.0, 0.0), 15: (5.0, 2.5), 30: (10.0, 5.0), 60: (20.0, 10.0)}
FIG_FILES = {
    "full (2019-01 .. 2026-08)": "equity_classical_full.png",
    "2021 (bull)": "equity_2021.png",
    "2022 (drawdown)": "equity_2022.png",
    "2023-01 .. 2026-08 (recovery)": "equity_2023_2026.png",
}


def strategies() -> list[tuple[str, object, str, int, bool]]:
    """(label, strategy instance, rebalance, warmup, is_benchmark). Fresh instances per call
    (EqualWeight/SpyReference keep an `_invested` flag)."""
    classical = [MeanReversion(), TrendFollowing(), TrendFollowing(sizing="slots"), WeightedTrend(), MomentumTopK()]
    rows = [(s.name, s, s.rebalance, s.warmup, False) for s in classical]
    rows += [
        ("EW-rebal (monthly)", EqualWeight(TICKERS, rebalance=True), "monthly", 0, True),
        ("EW-hold", EqualWeight(TICKERS, rebalance=False), "monthly", 0, True),
        ("SPY (reference)", SpyReference(), "monthly", 0, True),
    ]
    return rows


def slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_")


def trim(res: BacktestResult, start: pd.Timestamp) -> BacktestResult:
    """Cut the pre-period (all-cash) bars off a result so metrics see only the period."""
    return replace(
        res,
        equity=res.equity.loc[start:], cash=res.cash.loc[start:], weights=res.weights.loc[start:],
        turnover=res.turnover.loc[start:], trades=[f for f in res.trades if f.date >= start],
        config={**res.config, "period_start": str(start.date()), "n_bars": int((res.equity.index >= start).sum())},
    )


def run_period(strategy, market: MarketData, start: str | None, end: str | None,
               rebalance: str, warmup: int, cost: CostModel) -> BacktestResult:
    dates = market.dates
    first = int(dates.searchsorted(pd.Timestamp(start))) if start else 0
    lo = max(first - warmup, 0)
    m = market.slice(dates[lo], pd.Timestamp(end) if end else None)
    cfg = BacktestConfig(rebalance=rebalance, warmup=first - lo, cost_model=cost)  # type: ignore[arg-type]
    res = run_backtest(strategy, m, cfg)
    return trim(res, dates[first])


def run_all(market: MarketData):
    out: dict[str, dict[str, dict[str, BacktestResult]]] = {}
    for period, (start, end) in PERIODS.items():
        out[period] = {}
        for cost_name, cost in COSTS.items():
            runs = {}
            for label, strat, reb, warm, _ in strategies():
                r = run_period(strat, market, start, end, reb, warm, cost)
                runs[label] = r
                r.save(RUNS / slug(period) / slug(cost_name), name=slug(label))
            out[period][cost_name] = runs
            print(f"  ran {period} / {cost_name}")
    return out


def cost_sensitivity(market: MarketData) -> pd.DataFrame:
    start, end = PERIODS["full (2019-01 .. 2026-08)"]
    rows = {}
    for label, strat, reb, warm, is_bm in strategies():
        if label == "SPY (reference)":
            continue
        rows[label] = {}
        for rt, (c, s) in SENSITIVITY_RT_BPS.items():
            fresh = next(x for x in strategies() if x[0] == label)[1]
            r = run_period(fresh, market, start, end, reb, warm, CostModel(c, s))
            rows[label][f"{rt} bps"] = cagr(r.equity)
        rows[label]["turnover/yr"] = compute_metrics(r)["Turnover"]
    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "CAGR at round-trip cost"
    return df


def plot_period(runs: dict[str, BacktestResult], title: str, path: Path) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    colors = {"MeanRev": "#d62728", "Trend(SMA50>SMA200), 1/N": "#17becf", "Trend": "#1f77b4", "WeightedTrend": "#9467bd", "MomTop5": "#ff7f0e",
              "EW-rebal": "#2ca02c", "EW-hold": "#8c564b", "SPY": "#7f7f7f"}
    for label, res in runs.items():
        key = next(k for k in colors if label.startswith(k))
        bench = key in ("EW-rebal", "EW-hold", "SPY")
        ax.plot(res.equity.index, res.equity / res.equity.iloc[0], color=colors[key],
                ls="--" if bench else "-", lw=1.2 if bench else 1.5, label=label)
    ax.set_yscale("log")
    lo, hi = min(r.equity.min() / r.equity.iloc[0] for r in runs.values()), max(r.equity.max() / r.equity.iloc[0] for r in runs.values())
    ticks = [t for t in (0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.25, 1.5, 2, 3, 4, 5, 6, 8) if lo * 0.95 <= t <= hi * 1.05] or [1]
    ax.set_yticks(ticks)
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.FormatStrFormatter("%gx"))
    ax.get_yaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_ylabel("growth of $1 (log scale)")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def write_report(results, sens: pd.DataFrame, market: MarketData) -> Path:
    REPORTS.mkdir(parents=True, exist_ok=True)
    md = ["# M3 results: classical strategies vs equal-weight buy-and-hold", ""]
    md += [
        f"Generated by `scripts/run_classical.py` from the committed snapshot ({market.dates[0].date()} .. "
        f"{market.dates[-1].date()}). Engine: decisions at close[t], fills at open[t+1], integer shares, no "
        "shorting/leverage, initial cash $1,000,000. Sharpe/Sortino use **rf = 0**; 252 bars/year. Costs = 10 bps "
        "commission + 5 bps slippage per side (30 bps round trip). Universe = 15 US large-caps chosen with hindsight "
        "(**survivorship bias**, decisions.md #2); SPY is a reference line, not tradeable.",
        "",
        "**Parameter honesty:** every strategy runs on textbook defaults (MeanRev N=20, z<-1; Trend SMA50>SMA200; "
        "WeightedTrend = blend of 1/3/6/12-month returns clipped at 0, 25% cap; MomTop5 = 12-1 month, k=5). "
        "Nothing was tuned on any reporting period, and no walk-forward tuning was done at all for M3 "
        "(decisions.md #19). The results below are therefore *first-run* numbers, not best-of-N.",
        "",
        "**Period handling:** each run is entered from cash on the period's first bar and is given its warmup "
        "(up to 252 bars) of *pre-period* history so the signal is live from day one; equity is reported from the "
        "period's first bar (decisions.md #17/#20). \"full\" is 2019-01-02 .. 2026-08-14 so every strategy, "
        "including the 252-day ones, starts on the same bar (M2's benchmark report keeps 2018).",
        "",
        "Rebalance frequencies: MeanRev **weekly**, Trend **weekly**, WeightedTrend **monthly**, MomTop5 **monthly**, "
        "EW-rebal monthly. Each run's config (incl. strategy `params`), equity, weights and trades are saved under "
        "`reports/runs/<period>/<cost>/<strategy>/`.",
        "",
    ]
    for period, by_cost in results.items():
        md += [f"## {period}", "", f"![equity {period}](fig/{FIG_FILES[period]})", ""]
        for cost_name, runs in by_cost.items():
            md += [f"### {cost_name}", "", format_table(summary_table(runs, rf=0.0)), ""]

    md += ["## Cost sensitivity (full period CAGR vs round-trip cost)", "",
           "Same runs, costs swept at 0 / 15 / 30 / 60 bps round trip (commission:slippage 2:1 per side). "
           "Turnover is annualised one-way (sum |fill notional| / equity / 2).", ""]
    s = sens.copy()
    for c in s.columns:
        s[c] = s[c].map(lambda v: f"{v:.1%}" if c.endswith("bps") else f"{v:.2f}")
    md += [_md(s), ""]

    md += interpretation(results, sens)
    path = REPORTS / "M3-results.md"
    path.write_text("\n".join(md))
    return path


def _md(df: pd.DataFrame) -> str:
    from sab.metrics.metrics import _to_markdown
    return _to_markdown(df)


def interpretation(results, sens: pd.DataFrame) -> list[str]:
    """Prose with the numbers pulled from the results (so they cannot drift from the tables).
    The sentences describe the committed-snapshot run; if the data or defaults change, re-read
    them -- the numbers will update, the adjectives will not."""
    C, Z = "with costs (10+5 bps)", "no costs"
    full = results["full (2019-01 .. 2026-08)"]
    y21, y22, rec = results["2021 (bull)"], results["2022 (drawdown)"], results["2023-01 .. 2026-08 (recovery)"]
    M = lambda runs, name, key: compute_metrics(runs[name])[key]
    P = lambda v: f"{v:.1%}"
    names = list(full[C])
    ew, ewh = "EW-rebal (monthly)", "EW-hold"
    mr = next(n for n in names if n.startswith("MeanRev"))
    tf = next(n for n in names if n.startswith("Trend") and "slots" not in n)
    ts = next(n for n in names if "slots" in n)
    wt = next(n for n in names if n.startswith("WeightedTrend"))
    mk = next(n for n in names if n.startswith("MomTop"))
    return [
        "## Interpretation (honest version)", "",
        f"- **Full period (2019-2026), after costs:** the two momentum-weighted strategies beat the official benchmark on "
        f"CAGR -- WeightedTrend {P(M(full[C], wt, 'CAGR'))} and MomTop5 {P(M(full[C], mk, 'CAGR'))} vs EW-rebal "
        f"{P(M(full[C], ew, 'CAGR'))} -- but with more volatility ({P(M(full[C], wt, 'Vol'))} / {P(M(full[C], mk, 'Vol'))} vs "
        f"{P(M(full[C], ew, 'Vol'))}), so their Sharpe ({M(full[C], wt, 'Sharpe'):.2f} / {M(full[C], mk, 'Sharpe'):.2f}) is no better "
        f"than EW-rebal's {M(full[C], ew, 'Sharpe'):.2f}, and both trail EW-hold ({P(M(full[C], ewh, 'CAGR'))}, Sharpe "
        f"{M(full[C], ewh, 'Sharpe'):.2f}). Their edge over EW-rebal is the same thing EW-hold's is: holding more of the names "
        "we already know became giants. Read as \"no risk-adjusted alpha from classical rules on this universe\", not \"momentum works\".",
        f"- **Trend-following (SMA50>SMA200) is a sizing story.** Equal-weighting the trending names keeps the book ~100% invested "
        f"whenever anything trends and returned {P(M(full[C], tf, 'CAGR'))} (Sharpe {M(full[C], tf, 'Sharpe'):.2f}), i.e. EW-rebal "
        f"with extra turnover. The 1/N-slot version -- each name long or cash -- is the real trend *filter*: {P(M(full[C], ts, 'CAGR'))} "
        f"CAGR at {P(M(full[C], ts, 'Exposure'))} average exposure, vol {P(M(full[C], ts, 'Vol'))}, MaxDD {P(M(full[C], ts, 'MaxDD'))} "
        f"(EW-rebal {P(M(full[C], ew, 'MaxDD'))}), Sharpe {M(full[C], ts, 'Sharpe'):.2f}. Same risk-adjusted return, a third less "
        "drawdown, a lot less absolute return. On this sample the trend filter bought insurance, not alpha.",
        f"- **2022 drawdown, after costs:** EW-rebal {P(M(y22[C], ew, 'CAGR'))}, SPY {P(M(y22[C], 'SPY (reference)', 'CAGR'))}. "
        f"WeightedTrend {P(M(y22[C], wt, 'CAGR'))} and MomTop5 {P(M(y22[C], mk, 'CAGR'))} finished *positive*: both rotated into the "
        f"energy names (XOM/CVX) that trended up all year, while staying ~fully invested. Trend-equal {P(M(y22[C], tf, 'CAGR'))} at "
        f"{P(M(y22[C], tf, 'Exposure'))} exposure did the same rotation, less cleanly; Trend-slots {P(M(y22[C], ts, 'CAGR'))} at "
        f"{P(M(y22[C], ts, 'Exposure'))} exposure, vol {P(M(y22[C], ts, 'Vol'))}, MaxDD {P(M(y22[C], ts, 'MaxDD'))} is what de-risking "
        f"looks like. MeanRev {P(M(y22[C], mr, 'CAGR'))} bought every dip in a year when dips kept dipping -- the worst line in the table.",
        f"- **2021 bull, after costs:** EW-rebal {P(M(y21[C], ew, 'CAGR'))} and nothing beat it; Trend {P(M(y21[C], tf, 'CAGR'))}, "
        f"MomTop5 {P(M(y21[C], mk, 'CAGR'))} and WeightedTrend {P(M(y21[C], wt, 'CAGR'))} were within a couple of points, MeanRev "
        f"{P(M(y21[C], mr, 'CAGR'))} ({P(M(y21[Z], mr, 'CAGR'))} before costs). In a broad melt-up any rule that is not always fully "
        "invested, or that trades a lot, pays for it.",
        f"- **2023-2026 recovery, after costs:** EW-rebal {P(M(rec[C], ew, 'CAGR'))} (Sharpe {M(rec[C], ew, 'Sharpe'):.2f}); WeightedTrend "
        f"{P(M(rec[C], wt, 'CAGR'))} and MomTop5 {P(M(rec[C], mk, 'CAGR'))} edged it on CAGR with much worse Sharpe "
        f"({M(rec[C], wt, 'Sharpe'):.2f} / {M(rec[C], mk, 'Sharpe'):.2f}) and drawdown ({P(M(rec[C], wt, 'MaxDD'))} / "
        f"{P(M(rec[C], mk, 'MaxDD'))} vs {P(M(rec[C], ew, 'MaxDD'))}) -- concentrated bets on NVDA-led tech.",
        f"- **Costs are the whole story for the fast strategy.** MeanRev turns over {sens.loc[mr, 'turnover/yr']:.0f}x/yr and its "
        f"full-period CAGR goes {P(sens.loc[mr, '0 bps'])} -> {P(sens.loc[mr, '15 bps'])} -> {P(sens.loc[mr, '30 bps'])} -> "
        f"{P(sens.loc[mr, '60 bps'])} at 0 / 15 / 30 / 60 bps round trip (${M(full[C], mr, 'Costs'):,.0f} of costs on $1M). The weekly "
        f"trend rules turn over {sens.loc[tf, 'turnover/yr']:.1f}x / {sens.loc[ts, 'turnover/yr']:.1f}x per year and lose ~1-2 CAGR points "
        f"between 0 and 60 bps ({P(sens.loc[tf, '0 bps'])} -> {P(sens.loc[tf, '60 bps'])}); the monthly ones (WeightedTrend "
        f"{P(sens.loc[wt, '0 bps'])} -> {P(sens.loc[wt, '60 bps'])}, MomTop5 {P(sens.loc[mk, '0 bps'])} -> {P(sens.loc[mk, '60 bps'])}) "
        f"and EW-rebal ({P(sens.loc[ew, '0 bps'])} -> {P(sens.loc[ew, '60 bps'])}) barely notice.",
        f"- **Mean reversion, as specified, does not work here even for free** ({P(M(full[Z], mr, 'CAGR'))} CAGR at zero cost vs "
        f"{P(M(full[Z], ew, 'CAGR'))} for EW-rebal, at {P(M(full[Z], mr, 'Exposure'))} average exposure): a 20-day z-score on daily "
        "large-cap closes spends bull years buying whatever lagged and bear years catching falling knives. It stays in the table "
        "unchanged -- tuning N or the threshold on these years would be exactly the in-sample fitting decisions.md #19 forbids.",
        "- **Caveat -- survivorship:** all 15 names were picked in 2026 knowing they survived and (mostly) grew. Every absolute number "
        "here -- and especially the momentum strategies' edge, which comes from riding names we already know became giants -- is "
        "inflated relative to a point-in-time 2018 universe. Only the *relative* ordering under identical rules is meaningful, and "
        "even that favours long-only, always-invested rules on this sample.",
        "- Sub-period Turnover includes the initial entry from cash (0.5 one-way), as in M2. Broker cash-scaling and sell-capping counts "
        "are in each run's `config.json`.",
        "",
    ]


def main() -> None:
    market = load_market(list(TICKERS) + [BENCHMARK])
    RUNS.mkdir(parents=True, exist_ok=True)
    print("running strategies ...")
    results = run_all(market)
    print("cost sensitivity ...")
    sens = cost_sensitivity(market)
    for period, runs in results.items():
        plot_period(runs["with costs (10+5 bps)"], f"{period}, with costs (10+5 bps/side), growth of $1", FIG / FIG_FILES[period])
    report = write_report(results, sens, market)
    (RUNS / "INDEX.json").write_text(json.dumps(
        {p: {c: {n: r.config for n, r in runs.items()} for c, runs in bc.items()} for p, bc in results.items()}, indent=1))
    print(f"wrote {report}")
    print(format_table(summary_table(results["full (2019-01 .. 2026-08)"]["with costs (10+5 bps)"])))
    print(sens)


if __name__ == "__main__":
    main()
