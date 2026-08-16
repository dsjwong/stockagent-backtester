"""M2 benchmark: equal-weight buy-and-hold (monthly rebalance / no rebalance) + SPY reference,
over the full period, 2021 (bull) and 2022 (drawdown), with and without costs.

Writes reports/M2-benchmark.md and reports/fig/equity_full.png. Offline: reads only the
committed snapshots. Run:  .venv/bin/python scripts/run_benchmark.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import matplotlib.ticker
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sab.data.universe import BENCHMARK, TICKERS  # noqa: E402
from sab.engine import BacktestConfig, BacktestResult, DefaultCost, ZeroCost, load_market, run_backtest  # noqa: E402
from sab.metrics.metrics import format_table, summary_table, total_return  # noqa: E402
from sab.strategies.buy_and_hold import EqualWeight, SpyReference  # noqa: E402

REPORTS = ROOT / "reports"
FIG = REPORTS / "fig"
PERIODS = {"full": (None, None), "2021 (bull)": ("2021-01-01", "2021-12-31"), "2022 (drawdown)": ("2022-01-01", "2022-12-31")}
COSTS = {"with costs (10+5 bps)": DefaultCost, "no costs": ZeroCost}


def run_all() -> dict[str, dict[str, dict[str, BacktestResult]]]:
    market = load_market(list(TICKERS) + [BENCHMARK])
    out: dict[str, dict[str, dict[str, BacktestResult]]] = {}
    for period, (start, end) in PERIODS.items():
        m = market.slice(pd.Timestamp(start) if start else None, pd.Timestamp(end) if end else None)
        out[period] = {}
        for cost_name, cost in COSTS.items():
            cfg = BacktestConfig(rebalance="monthly", warmup=0, cost_model=cost)
            out[period][cost_name] = {
                "EW-15 rebal (monthly)": run_backtest(EqualWeight(TICKERS, rebalance=True), m, cfg, name="EW-rebal"),
                "EW-15 hold (no rebal)": run_backtest(EqualWeight(TICKERS, rebalance=False), m, cfg, name="EW-hold"),
                "SPY (reference)": run_backtest(SpyReference(), m, cfg, name="SPY"),
            }
    return out


def sanity_check(results, market_slice) -> str:
    """EW-hold at zero cost should ~= mean of individual ticker total returns, measured from
    the first fill (open of bar 1) to the last close -- the only differences are integer
    shares (idle cash) and that equity[0] is cash marked at close[0], not open[1]."""
    res = results["no costs"]["EW-15 hold (no rebal)"]
    strat_ret = total_return(res.equity)
    per_ticker = market_slice.close.iloc[-1][list(TICKERS)] / market_slice.open.iloc[1][list(TICKERS)] - 1
    mean_ret = float(per_ticker.mean())
    idle = float(res.cash.iloc[-1] / res.equity.iloc[-1])
    lines = [
        f"- EW-hold (zero cost) total return: **{strat_ret:.2%}**",
        f"- Mean of the 15 tickers' total returns (open of first fill bar -> last close): **{mean_ret:.2%}**",
        f"- Difference: {strat_ret - mean_ret:+.2%} (idle cash at end: {idle:.3%} of equity; integer-share "
        "rounding at the initial buy leaves a little cash uninvested, and equity starts at cash marked "
        "on bar 0 while the buys happen at bar 1's open).",
        "- Per-ticker total returns: " + ", ".join(f"{t} {v:+.0%}" for t, v in per_ticker.sort_values(ascending=False).items()),
    ]
    return "\n".join(lines)


def plot_full(results_full: dict[str, dict[str, BacktestResult]]) -> Path:
    FIG.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    styles = {"EW-15 rebal (monthly)": ("#1f77b4", "-"), "EW-15 hold (no rebal)": ("#2ca02c", "-"), "SPY (reference)": ("#7f7f7f", "--")}
    for cost_name, ls_alpha in (("with costs (10+5 bps)", 1.0), ("no costs", 0.35)):
        for name, res in results_full[cost_name].items():
            color, ls = styles[name]
            label = f"{name}" + ("" if ls_alpha == 1.0 else " (no costs)")
            ax.plot(res.equity.index, res.equity / res.equity.iloc[0], color=color, ls=ls, alpha=ls_alpha, lw=1.4, label=label)
    ax.set_yscale("log")
    ax.set_yticks([1, 1.5, 2, 3, 4, 5, 6, 7])
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.FormatStrFormatter("%gx"))
    ax.get_yaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_ylabel("growth of $1 (log scale)")
    ax.set_title("Equal-weight buy-and-hold of the 15-name universe vs SPY, 2018-01 .. 2026-08")
    ax.grid(True, which="both", alpha=0.25)
    ax.axvspan(pd.Timestamp("2021-01-01"), pd.Timestamp("2021-12-31"), color="#2ca02c", alpha=0.06)
    ax.axvspan(pd.Timestamp("2022-01-01"), pd.Timestamp("2022-12-31"), color="#d62728", alpha=0.06)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    path = FIG / "equity_full.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def write_report(results, market) -> Path:
    REPORTS.mkdir(parents=True, exist_ok=True)
    md = ["# M2 benchmark: equal-weight buy-and-hold vs SPY", ""]
    md += [
        "Generated by `scripts/run_benchmark.py` from the committed snapshot "
        f"({market.dates[0].date()} .. {market.dates[-1].date()}, {len(market)} bars). "
        "Engine: decisions at close[t], fills at open[t+1], integer shares, no shorting, "
        "monthly rebalance (first bar of the month), initial cash $1,000,000. "
        "Sharpe/Sortino use **rf = 0**; 252 trading days/year. Costs = 10 bps commission + 5 bps "
        "slippage per side. Universe chosen with hindsight (survivorship bias, see decisions.md #2).",
        "",
        "![equity curves](fig/equity_full.png)",
        "",
    ]
    for period, by_cost in results.items():
        md.append(f"## {period}")
        for cost_name, runs in by_cost.items():
            md += [f"### {cost_name}", "", format_table(summary_table(runs, rf=0.0)), ""]
    full = results["full"]
    md += ["## Sanity check: EW-hold vs mean of individual returns", "", sanity_check(full, market), ""]

    ew_c = full["with costs (10+5 bps)"]["EW-15 rebal (monthly)"]
    ew_0 = full["no costs"]["EW-15 rebal (monthly)"]
    md += [
        "## Notes",
        "",
        f"- Cost drag on EW-rebal (monthly) over the full period: ${sum(f.total_cost for f in ew_c.trades):,.0f} paid "
        f"on $1M initial across {len(ew_c.trades)} fills; total return {total_return(ew_c.equity):.1%} with costs vs "
        f"{total_return(ew_0.equity):.1%} without. Costs barely matter for a monthly-rebalanced portfolio; they will "
        "matter for anything that trades daily (M3/M4).",
        "- EW-rebal underperforms EW-hold over the full period (19.8% vs 24.0% CAGR): monthly rebalancing keeps "
        "trimming the runaway winner (NVDA +4,368%) into laggards. It is the better-behaved benchmark (lower vol, "
        "no single-name concentration) and stays the official one; EW-hold is reported alongside.",
        "- `Turnover` counts the initial entry from cash (0.5 one-way), so 1-year runs show >= 0.5 by construction; "
        "over the full period it amortises to 0.06/yr for the hold variants.",
        f"- Broker cash-scaling events (buys trimmed because a gap-down at the open left less cash than sized at the "
        f"previous close): {ew_c.config['n_scaled_down']} of {sum(1 for _ in set(f.date for f in ew_c.trades))} rebalance fills. "
        "This is the conservative side of sizing off decision-time equity; the portfolio never borrows.",
        "- Sub-period runs start from cash on the first bar of the period and buy at the second bar's open, "
        "so `Exposure` < 100% by one bar and 2021/2022 numbers are for a portfolio *entered* that year, not a slice of the full run.",
        "- EW-hold has no sells, so `WinRate` is undefined (blank). SPY is 100% one ETF and not part of the tradeable universe.",
        "",
    ]
    path = REPORTS / "M2-benchmark.md"
    path.write_text("\n".join(md))
    return path


def main() -> None:
    results = run_all()
    market = load_market(list(TICKERS) + [BENCHMARK])
    fig = plot_full(results["full"])
    report = write_report(results, market)
    print(f"wrote {report} and {fig}")
    print(format_table(summary_table(results["full"]["with costs (10+5 bps)"])))


if __name__ == "__main__":
    main()
