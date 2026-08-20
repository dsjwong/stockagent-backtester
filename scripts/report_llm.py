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
        held = [None if a is None else sorted(k for k, v in a.items() if v > 1e-6) for a in j.get("answers_real", [])]
        details.append(f"- **{j['provider']}/{j['model']} {j['mode']} @ {j['date']}, T={j['temperature']}, n={j['n']}**: held sets per draw: "
                       + "; ".join("unparseable" if h is None else ("[all cash]" if not h else f"[{', '.join(h)}]") for h in held)
                       + f". First rationale: \"{(j.get('texts') or [''])[0].split('rationale')[-1].strip(' \":}').strip()[:200]}\"")
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
    out_md = REPORTS / "M4-llm-results.SMOKE.md" if INCLUDE_MOCK else OUT_MD   # never overwrite the real report with a mock one
    out_md.write_text("\n".join(parts))
    print(f"wrote {out_md}")
    for b in rva_blocks:
        print(b)


# --------------------------------------------------------------------------- hand-written prose (after the runs)
INTERPRETATION.update({
    "header": (
        "Generated by `scripts/report_llm.py` from the saved runs under `reports/runs/llm/` (no model calls). "
        "Agent: `LLMAgent` (decisions.md #22-#25) on **gemini-3.5-flash-lite** (Gemini API free tier, REST, thinking `minimal`, "
        "temperature 0, seed 0, `max_tokens` 400), weekly decisions at the close with fills at the next open, 60-bar price "
        "window + 1w/1m/3m/6m trailing returns + portfolio state in the prompt, long-only, sum of weights <= 1. Two prompt modes: "
        "**anon** (tickers -> `Stock 01..15` via a seeded shuffle, dates -> `Day t`, prices rebased to 100) and **real** (symbols, "
        "calendar dates, adjusted closes). Same engine, costs (10 bps commission + 5 bps slippage per side), period handling "
        "(entered from cash on the period's first bar with pre-period warmup history) and benchmarks as `reports/M3-results.md`; "
        "rf = 0. Every prompt/response pair is in `data/llm_cache/` (committed), so the tables replay with zero API calls. "
        "Model choice and quota handling: decisions.md #27-#28 (the planned `gemini-2.5-flash` is gone for new users; the "
        "flash models are capped at 20 requests/day on the free tier, flash-lite at 500).\n"
    ),
    "consistency": (
        "Reading: the consistency check starts from an all-cash portfolio on 2022-06-13 (the Monday of the June-2022 selloff week) so "
        "it is the same prompt every time, not the state-dependent prompt of the backtest. Anonymised, the model gave the **same "
        "four names (XOM, CVX, JNJ, KO) in all 8 draws** and only moved weights around by 0.1 (5 distinct weight vectors at T=0.7, "
        "2 at T=0 -- so T=0 without a seed is *not* bit-identical on this API; the main runs pass `seed=0` and are cached, so they "
        "are reproducible regardless). With real names the answer was **100% cash in all 8 draws** with a \"broad market selloff\" "
        "rationale. Sampling noise is therefore small relative to the prompt-mode effect: the held set is stable at T=0.7, but "
        "whether the model is invested at all depended on whether it knew the date and the tickers.\n"
    ),
    "real_vs_anon": (
        "Naming the tickers and dates changes behaviour and results. Same data, same engine, same seed: in 2022 the real-mode agent "
        "returned **+12.0%** vs **+1.6%** anonymised (no costs: 13.5% vs 4.3%), with half the turnover (4.5x vs 8.6x/yr), one more name "
        "held on average and a lower drawdown (-15.5% vs -21.6%); in 2021 it was the other way round in return (32.9% vs 41.5%) with "
        "similar Sharpe (2.37 vs 2.29). Across (date, ticker) the two weight paths correlate at only 0.63-0.64, the average target "
        "portfolios differ by ~0.37-0.39 in half-L1 distance and share ~53% of their held names (Jaccard), although the equity curves "
        "still correlate at ~0.88-0.89 because both chase the same trailing returns. The qualitative difference is in the rationales: "
        "the anonymised agent talks only about \"momentum leaders\" and \"medium-term trends\" (it can see nothing else), while the real-"
        "mode agent talks about \"defensive energy and healthcare\", \"high-beta tech drawdowns\" and an \"ongoing correction\" -- sector "
        "labels and a regime narrative that are not in the prompt -- and held zero big-tech from Q2 2022 and 71% XOM+CVX in Q4. The "
        "cleanest evidence is the consistency probe above: on 2022-06-13, from cash, the real-mode agent went 100% cash in 8/8 draws, the "
        "anonymised agent 100% invested in 8/8 draws. None of this proves the model *remembers* 2022 price paths, but it does show that "
        "world knowledge (sector membership, what 2022 was like) leaks into real-mode decisions, which is why only the anon numbers are "
        "read as \"what the model does with the data\" and the real numbers as an upper bound contaminated by hindsight. (The no-future "
        "prompt audit and the perturbation tests rule out leakage *through the prompt*; they cannot rule out leakage through the weights "
        "of the model, which is exactly what anonymisation is for.)\n"
    ),
    "interpretation": (
        "1. The anonymised Gemini agent is, in practice, a discretionary momentum strategy: nearly every rationale says \"momentum "
        "leaders\", it holds 5-7 names (often 2-4 in early 2022, with 64% in XOM+CVX in Q1 2022), and it turns the book over 8.6-9.3x a "
        "year, which costs 2.6-3.4% a year at 30 bps round trip (the `[no costs]` rows). "
        "2. In the 2022 drawdown it beat EW-rebal after costs (+1.6% vs -9.7%, Sharpe 0.18 vs -0.35) -- but so did every momentum "
        "rule in M3 (WeightedTrend +3.0%, MomTop5 +1.7%, Trend -3.3%), and it did so with a deeper drawdown than any of them "
        "(-21.6% vs -14.2% WeightedTrend, -8.8% Trend slots); its 2022 edge is the energy trade that the trailing-return block hands "
        "to anyone who reads it. "
        "3. In the 2021 bull year it trailed EW-rebal (41.5% vs 42.3% CAGR; Sharpe 2.29 vs 2.97; MaxDD -7.4% vs -4.1%) and both "
        "trend rules on Sharpe; the turnover it paid for bought nothing. "
        "4. Chained over 2021-2022 the anon agent made +43.5% vs +28.4% for EW-rebal, essentially all of it from 2022; this is two "
        "years, one model, one seed, 104 decisions -- not evidence of skill, and a Sharpe of 0.18 in the good year is not a strategy. "
        "5. Sharpe ranks the anon LLM below EW-rebal in 2021 and below WeightedTrend in 2022, and its drawdowns are the worst of the "
        "non-MeanRev strategies in both years, so on risk-adjusted terms it does not beat the cheap rules after costs. "
        "6. The real-mode agent looks better (2022: +12.0%, Sharpe 0.70, the best line in the table) but the real-vs-anon section "
        "shows its decisions use knowledge that is not in the prompt; treat it as contaminated. "
        "7. Engineering-wise the pipeline held up: 208 decisions, 0 parse failures, 0 retries, 0 fallbacks, 1 unknown name dropped, "
        "no renormalisations; the model omits some tickers in 15-35 of 52 answers (allowed by the spec, counted anyway). "
        "8. What would change the conclusion: more seeds/years, a second model, and "
        "a prompt that does not hand the model trailing returns -- the current prompt makes \"momentum\" the path of least resistance.\n"
    ),
    "cost_tail": (
        "$0 on the free tier; plus 16 uncached consistency calls and ~25 probe/partial calls (17 cached decisions of a `gemini-3.6-flash` "
        "2022-anon run that is resumable at 20 requests/day). Each run took 6-20 minutes wall-clock at the paced ~13 requests/min "
        "(latency column includes 429 backoff waits). Re-running any table here costs nothing: the cache is committed.\n\n"
        "**Not done / how to extend.** The full-period 2019-2026 anon run needed two free-tier days (~397 calls against a 500/day "
        "cap) and is complete and cached. Not run: full-period **real** mode "
        "(`.venv/bin/python scripts/run_llm.py --provider gemini --model gemini-3.5-flash-lite --mode real --period full --verbose`) "
        "and a second model (`--model gemini-3.6-flash --period 2022`, 20 calls/day, 17 decisions already cached). Either exits 2 "
        "with a resume hint if the daily quota runs out; re-running the same command resumes from the cache, then "
        "`.venv/bin/python scripts/report_llm.py` adds the period here."
    ),
})

if __name__ == "__main__":
    main()
