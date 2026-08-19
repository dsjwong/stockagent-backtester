"""M5: build reports/M4-llm-results.md + reports/fig/llm_<period>.png from the SAVED LLM runs
(reports/runs/llm/...) and the saved M3 classical runs (reports/runs/<period>/...). Offline, no
model calls, no re-runs -- except the zero-cost *replay* of each LLM agent's logged decisions
(same weights on the same dates, ZeroCost), which costs nothing and needs no key.

    .venv/bin/python scripts/report_llm.py

Sections per period: metrics table (LLM agents with costs, the same decisions replayed at zero
cost, EW-rebal / EW-hold / SPY / Trend x2 / WeightedTrend / MeanRev with costs), equity + exposure
figure, per-agent decision stats (exposure, names held, turnover, calls/tokens, retries /
fallbacks / spec repairs), consistency-check results, sample rationales, real-vs-anon comparison.
The interpretation paragraphs live in INTERPRETATION below (hand-written after reading the
numbers; re-running the script keeps them).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_classical import PERIODS, run_period, slug  # noqa: E402
from sab.data.universe import BENCHMARK, TICKERS  # noqa: E402
from sab.engine import BacktestResult, ZeroCost, load_market  # noqa: E402
from sab.engine.backtest import Context  # noqa: E402
from sab.engine.types import Order  # noqa: E402
from sab.metrics.metrics import compute_metrics, format_table, summary_table  # noqa: E402
from sab.strategies._base import target_orders  # noqa: E402

REPORTS = ROOT / "reports"
FIG = REPORTS / "fig"
RUNS_LLM = REPORTS / "runs" / "llm"
RUNS_CLASSICAL = REPORTS / "runs"
COST_DIR = "with_costs_10_5_bps"
OUT_MD = REPORTS / "M4-llm-results.md"

PERIOD_ORDER = ["2022 (drawdown)", "2021 (bull)", "full (2019-01 .. 2026-08)", "2023-01 .. 2026-08 (recovery)"]
BENCH_SLUGS = [("EW-rebal (monthly)", "EW-rebal_monthly"), ("EW-hold", "EW-hold"), ("SPY (reference)", "SPY_reference"),
               ("Trend(SMA50>SMA200)", "Trend_SMA50_SMA200"), ("Trend(SMA50>SMA200), 1/N slots", "Trend_SMA50_SMA200_1_N_slots"),
               ("WeightedTrend(21/63/126/252)", "WeightedTrend_21_63_126_252")]
MEANREV = ("MeanRev(N=20,z<-1)", "MeanRev_N_20_z_-1")
SPEC_REPAIRS = ("unknown_dropped", "non_numeric_dropped", "negative_clipped", "renormalised", "trailing_comma_fixed", "fence_stripped")
FMT_REPAIRS = ("fence_stripped", "trailing_comma_fixed")
SHORT = {"EW-rebal (monthly)": "EW-rebal", "EW-hold": "EW-hold", "SPY (reference)": "SPY", "Trend(SMA50>SMA200)": "Trend(equal)",
         "Trend(SMA50>SMA200), 1/N slots": "Trend(slots)", "WeightedTrend(21/63/126/252)": "WeightedTrend", "MeanRev(N=20,z<-1)": "MeanRev"}


# --------------------------------------------------------------------------- replay at zero cost

class ReplayWeights:
    """Re-emit the logged target weights of an LLM run on the same dates (ZeroCost counterfactual)."""

    def __init__(self, decisions: pd.DataFrame, name: str) -> None:
        self.by_date = {pd.Timestamp(d): w for d, w in decisions["weights"].items()}
        self.name = name
        self.last: dict[str, float] = {}

    def on_bar(self, ctx: Context) -> list[Order]:
        w = self.by_date.get(pd.Timestamp(ctx.date))
        if w is None:                         # not a logged decision bar: hold
            return []
        self.last = w
        return target_orders(ctx, w, list(TICKERS))


def replay_no_cost(res: BacktestResult, market, period_label: str) -> BacktestResult:
    start, end = PERIODS[period_label]
    agent_cfg = res.config.get("params") or res.config.get("strategy", {}).get("params") or {}
    reb = agent_cfg.get("rebalance", "weekly")
    warm = int(agent_cfg.get("warmup", 127))
    r = run_period(ReplayWeights(res.decisions, res.name + " [no costs]"), market, start, end, reb, warm, ZeroCost)
    r.name = res.name + " [no costs]"
    return r


# --------------------------------------------------------------------------- loading

INCLUDE_MOCK = "--include-mock" in sys.argv        # smoke-test the report on the committed mock run


def agent_runs() -> dict[str, list[tuple[Path, BacktestResult, dict[str, Any]]]]:
    out: dict[str, list] = {}
    for pdir in sorted(RUNS_LLM.iterdir()) if RUNS_LLM.exists() else []:
        if pdir.name == "consistency" or not (pdir / COST_DIR).exists():
            continue
        for d in sorted((pdir / COST_DIR).iterdir()):
            if d.is_dir() and d.name.startswith("LLM_") and (INCLUDE_MOCK or "mock" not in d.name) and (d / "summary.json").exists():
                res = BacktestResult.load(d)
                summary = json.loads((d / "summary.json").read_text())
                out.setdefault(summary["period"], []).append((d, res, summary))
    return out


def load_bench(period_label: str, label: str, sl: str, llm_dir: Path) -> BacktestResult | None:
    for base in (llm_dir, RUNS_CLASSICAL / slug(period_label) / COST_DIR):
        d = base / sl
        if (d / "equity.parquet").exists():
            r = BacktestResult.load(d)
            r.name = label
            return r
    return None


# --------------------------------------------------------------------------- stats

def agent_stats(res: BacktestResult, summary: dict[str, Any]) -> dict[str, Any]:
    dec = res.decisions
    reps = pd.DataFrame([r if isinstance(r, dict) else {} for r in dec["repairs"]]).reindex(columns=list(SPEC_REPAIRS) + ["missing_zeroed"]).fillna(0)
    m = compute_metrics(res)
    return {
        "decisions": len(dec), "model calls": int(dec["n_calls"].sum()), "cache hits (this run)": int(dec["cache_hit"].sum()),
        "prompt tokens": int(dec["prompt_tokens"].sum()), "completion tokens": int(dec["completion_tokens"].sum()),
        "mean exposure": float(dec["exposure"].mean()), "min exposure": float(dec["exposure"].min()), "max exposure": float(dec["exposure"].max()),
        "mean names held": float(dec["n_held"].mean()), "turnover/yr": m["Turnover"],
        "retries": int(dec["retried"].sum()), "fallbacks": int(dec["fallback"].sum()),
        "decisions with a spec repair": int((reps[list(SPEC_REPAIRS)].sum(axis=1) > 0).sum()),
        "  of which only formatting (fence/comma)": int(((reps[list(FMT_REPAIRS)].sum(axis=1) > 0) & (reps[[c for c in SPEC_REPAIRS if c not in FMT_REPAIRS]].sum(axis=1) == 0)).sum()),
        "renormalised (sum>1)": int((reps["renormalised"] > 0).sum()), "unknown names dropped": int((reps["unknown_dropped"] > 0).sum()),
        "decisions omitting some tickers (-> 0)": int((reps["missing_zeroed"] > 0).sum()),
        "mean latency (s)": float(dec["latency_s"].mean()),
    }


def weight_changes(dec: pd.DataFrame) -> pd.Series:
    """Mean absolute target-weight change between consecutive decisions (half L1 = 'intended turnover')."""
    ws = pd.DataFrame(list(dec["weights"])).reindex(columns=list(TICKERS)).fillna(0.0)
    return (ws.diff().abs().sum(axis=1) / 2).iloc[1:]


def real_vs_anon(a: BacktestResult, b: BacktestResult) -> dict[str, Any]:
    da, db = a.decisions, b.decisions
    common = da.index.intersection(db.index)
    wa = pd.DataFrame(list(da.loc[common, "weights"])).reindex(columns=list(TICKERS)).fillna(0.0)
    wb = pd.DataFrame(list(db.loc[common, "weights"])).reindex(columns=list(TICKERS)).fillna(0.0)
    held_a, held_b = wa.gt(1e-6), wb.gt(1e-6)
    jacc = ((held_a & held_b).sum(axis=1) / (held_a | held_b).sum(axis=1).replace(0, np.nan)).mean()
    corr = np.corrcoef(wa.to_numpy().ravel(), wb.to_numpy().ravel())[0, 1]
    l1 = (wa - wb).abs().sum(axis=1).mean() / 2
    ra, rb = a.equity.pct_change().dropna(), b.equity.pct_change().dropna()
    top_a = wa.mean().sort_values(ascending=False).head(5)
    top_b = wb.mean().sort_values(ascending=False).head(5)
    return {"n_common_decisions": len(common), "mean_exposure_real": float(da["exposure"].mean()), "mean_exposure_anon": float(db["exposure"].mean()),
            "weight_corr": float(corr), "mean_half_L1_distance": float(l1), "mean_jaccard_held": float(jacc),
            "daily_return_corr": float(ra.corr(rb)), "top5_real": {k: round(float(v), 3) for k, v in top_a.items()},
            "top5_anon": {k: round(float(v), 3) for k, v in top_b.items()}}


# --------------------------------------------------------------------------- figure

def plot_period(period_label: str, agents: list[BacktestResult], benches: dict[str, BacktestResult], path: Path) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(10, 7.5), sharex=True, gridspec_kw={"height_ratios": [3, 1.2]})
    agent_colors = ["#d62728", "#ff7f0e", "#e377c2", "#bcbd22"]
    bench_colors = {"EW-rebal": "#2ca02c", "EW-hold": "#8c564b", "SPY": "#7f7f7f", "Trend(equal)": "#1f77b4",
                    "Trend(slots)": "#17becf", "WeightedTrend": "#9467bd", "MeanRev": "#c7c7c7"}
    for i, res in enumerate(agents):
        eq = res.equity / res.equity.iloc[0]
        ax.plot(eq.index, eq, color=agent_colors[i % len(agent_colors)], lw=2.2, label=res.name, zorder=5)
        ax2.plot(res.exposure.index, res.exposure, color=agent_colors[i % len(agent_colors)], lw=1.5, label=res.name)
    for label, res in benches.items():
        key = SHORT.get(label, label)
        eq = res.equity / res.equity.iloc[0]
        ax.plot(eq.index, eq, color=bench_colors.get(key, "#999999"), lw=1.2 if key in ("EW-rebal", "EW-hold", "SPY") else 1.0,
                ls="-" if key in ("EW-rebal", "EW-hold", "SPY") else "--", label=key, alpha=0.95)
    ax.set_title(f"{period_label}: LLM agent(s) vs benchmarks (with costs 10+5 bps/side, growth of $1)")
    ax.set_ylabel("growth of $1")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2, loc="best")
    ax2.set_ylabel("LLM exposure")
    ax2.set_ylim(0, 1.05)
    ax2.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- prose

INTERPRETATION = {
    "intro": "",        # filled in below after the runs (hand-written, see bottom of file)
}


def md_table(df: pd.DataFrame, index: bool = True) -> str:
    """Markdown table without the tabulate dependency."""
    cols = ([df.index.name or ""] if index else []) + [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for idx, row in df.iterrows():
        cells = ([str(idx)] if index else []) + ["" if (isinstance(v, float) and np.isnan(v)) else str(v) for v in row.tolist()]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def fmt_pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def consistency_section() -> str:
    d = RUNS_LLM / "consistency"
    if not d.exists():
        return "_No consistency checks saved yet._\n"
    rows = []
    details = []
    for f in sorted(d.glob("*.json")):
        j = json.loads(f.read_text())
        rows.append({"agent": f"{j['provider']}/{j['model']} {j['mode']}", "date": j["date"], "T": j["temperature"], "n": j["n"],
                     "distinct answers": j["n_distinct"], "fraction identical": round(j["fraction_identical"], 2),
                     "exposures": ", ".join(f"{e:.2f}" for e in j.get("exposures", [])),
                     "mean |weight std|": (f"{j['weight_std_mean']:.3f}" if "weight_std_mean" in j else "n/a"),
                     "max weight range": (f"{j['weight_range_max']:.2f}" if "weight_range_max" in j else "n/a")})
        held = [sorted(k for k, v in a.items() if v > 1e-6) if a else None for a in j.get("answers_real", [])]
        details.append(f"- **{j['provider']}/{j['model']} {j['mode']} @ {j['date']}, T={j['temperature']}, n={j['n']}**: held sets per draw: "
                       + "; ".join("unparseable" if h is None else f"[{', '.join(h)}]" for h in held))
    df = pd.DataFrame(rows)
    return md_table(df, index=False) + "\n\n" + "\n".join(details) + "\n"


def rationales(res: BacktestResult, k: int = 4) -> list[str]:
    dec = res.decisions
    dec = dec[~dec["fallback"]]
    if len(dec) == 0:
        return []
    idx = np.linspace(0, len(dec) - 1, num=min(k, len(dec))).round().astype(int)
    out = []
    for i in idx:
        row = dec.iloc[int(i)]
        txt = str(row["rationale"]).strip().replace("\n", " ")
        out.append(f"- {pd.Timestamp(dec.index[int(i)]).date()} (exposure {row['exposure']:.2f}, {row['n_held']} names): \"{txt}\"")
    return out


def main() -> None:
    market = load_market(list(TICKERS) + [BENCHMARK])
    runs = agent_runs()
    if not runs:
        raise SystemExit("no real LLM runs under reports/runs/llm/*/with_costs_10_5_bps/LLM_* (mock runs are skipped)")
    parts: list[str] = []
    parts.append("# M4 results: LLM agent (Gemini) vs classical strategies and buy-and-hold\n")
    parts.append(INTERPRETATION["header"])
    totals = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "runs": 0}
    period_blocks: list[str] = []
    rva_blocks: list[str] = []
    for period in [p for p in PERIOD_ORDER if p in runs] + [p for p in runs if p not in PERIOD_ORDER]:
        entries = runs[period]
        llm_dir = entries[0][0].parent
        table_runs: dict[str, BacktestResult] = {}
        agents: list[BacktestResult] = []
        stats_rows: dict[str, dict[str, Any]] = {}
        for d, res, summary in entries:
            table_runs[res.name] = res
            agents.append(res)
            nc = replay_no_cost(res, market, period)
            table_runs[nc.name] = nc
            st = agent_stats(res, summary)
            st["turnover/yr (no-cost replay)"] = compute_metrics(nc)["Turnover"]
            st["CAGR with / without costs"] = f"{fmt_pct(compute_metrics(res)['CAGR'])} / {fmt_pct(compute_metrics(nc)['CAGR'])}"
            stats_rows[res.name] = st
            totals["calls"] += st["model calls"]; totals["prompt_tokens"] += st["prompt tokens"]; totals["completion_tokens"] += st["completion tokens"]
            totals["runs"] += 1
        benches: dict[str, BacktestResult] = {}
        for label, sl in BENCH_SLUGS + [MEANREV]:
            r = load_bench(period, label, sl, llm_dir)
            if r is not None:
                benches[label] = r
                table_runs[label] = r
        table = format_table(summary_table(table_runs, rf=0.0))
        fig_path = FIG / f"llm_{slug(period).split('_')[0] if not period.startswith('full') else 'full'}.png"
        plot_period(period, agents, benches, fig_path)
        blk = [f"## {period}\n", f"![equity {period}]({fig_path.relative_to(REPORTS).as_posix()})\n",
               "Top: growth of $1 with costs (LLM agents bold; `[no costs]` rows in the table replay the same logged weights at zero cost). Bottom: LLM target exposure (sum of weights) over time.\n",
               "### Metrics (costs 10+5 bps/side unless marked, rf = 0)\n", table, "",
               "### LLM decision statistics\n"]
        sdf = pd.DataFrame(stats_rows)
        sdf = sdf.map(lambda v: f"{v:.3f}" if isinstance(v, float) else v)
        blk.append(md_table(sdf) + "\n")
        for res in agents:
            blk.append(f"**Sample rationales, {res.name}** (verbatim, evenly spaced through the year):\n")
            blk.extend(rationales(res))
            blk.append("")
        if len(agents) >= 2:
            by_mode = {r.config.get("params", {}).get("mode") or r.name.split(", ")[-1].rstrip(")"): r for r in agents}
            if "real" in by_mode and "anon" in by_mode:
                cmp = real_vs_anon(by_mode["real"], by_mode["anon"])
                rva_blocks.append(f"**{period}**: {json.dumps(cmp, indent=None)}")
                blk.append("### Real vs anon on this period\n")
                blk.append(f"- common decision dates: {cmp['n_common_decisions']}; mean exposure real {cmp['mean_exposure_real']:.2f} vs anon {cmp['mean_exposure_anon']:.2f}\n"
                           f"- weight correlation across (date, ticker): {cmp['weight_corr']:.2f}; mean half-L1 distance between the two target portfolios: {cmp['mean_half_L1_distance']:.2f}; mean Jaccard overlap of held sets: {cmp['mean_jaccard_held']:.2f}\n"
                           f"- daily return correlation of the two equity curves: {cmp['daily_return_corr']:.2f}\n"
                           f"- top-5 average weights, real: {cmp['top5_real']}; anon: {cmp['top5_anon']}\n")
        period_blocks.append("\n".join(blk))
    parts.extend(period_blocks)
    parts.append("## Consistency check (same prompt, repeated calls, cache bypassed)\n")
    parts.append(consistency_section())
    parts.append(INTERPRETATION["consistency"])
    parts.append("## Real vs anonymised prompts\n")
    parts.append(INTERPRETATION["real_vs_anon"])
    parts.append("## Interpretation\n")
    parts.append(INTERPRETATION["interpretation"])
    parts.append("## Cost / usage\n")
    parts.append(f"{totals['runs']} agent-period runs in this report, {totals['calls']} model calls, {totals['prompt_tokens']:,} prompt + "
                 f"{totals['completion_tokens']:,} completion tokens (Gemini's count; ~3.4k prompt tokens per call because every price digit tokenises separately). "
                 + INTERPRETATION["cost_tail"])
    if INCLUDE_MOCK:
        OUT_MD = REPORTS / "M4-llm-results.SMOKE.md"   # never overwrite the real report with a mock one
    OUT_MD.write_text("\n".join(parts))
    print(f"wrote {OUT_MD}")
    for b in rva_blocks:
        print(b)


# --------------------------------------------------------------------------- hand-written prose (after the runs)
INTERPRETATION.update({
    "header": "PLACEHOLDER_HEADER\n",
    "consistency": "PLACEHOLDER_CONSISTENCY\n",
    "real_vs_anon": "PLACEHOLDER_RVA\n",
    "interpretation": "PLACEHOLDER_INTERP\n",
    "cost_tail": "PLACEHOLDER_COST",
})

if __name__ == "__main__":
    main()
