"""Build reports/forward.md (+ reports/fig/forward.png) from forward/decisions.jsonl and the stored bars.
Never calls a model. Equity of every arm comes from the historical engine (next-open fills, 10+5 bps).

    .venv/bin/python scripts/forward_report.py [--forward-dir forward] [--out reports/forward.md]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sab.data.universe import TICKERS  # noqa: E402
from sab.forward import DecisionLog, ForwardConfig, ReplayStrategy, load_forward_market  # noqa: E402
from sab.forward.analysis import MIN_DAYS_FOR_CI, bootstrap_gap, gap_returns  # noqa: E402
from sab.forward.runner import replay_arm  # noqa: E402
from sab.metrics.metrics import format_table, summary_table  # noqa: E402
from sab.strategies import EqualWeight, SpyReference  # noqa: E402

HYPOTHESIS = """\
## Hypothesis (stated before any forward decision existed)

The historical study found that the same model, given real tickers and dates, behaved very differently from
the same model given anonymised prices (2022: +11.9% real vs +1.6% anon; 8/8 vs 0/8 all-cash draws on
2022-06-13), and read that gap as **leakage**: the model remembers the period. After the model's training
cutoff there is nothing to remember. So:

* **H1 (leakage)**: if the historical gap was leakage, the forward real-minus-anon gap should **collapse
  toward zero** (and its sign/size should stop being systematic).
* **H0 (not only leakage)**: if the gap persists forward, the real prompt changes behaviour for reasons
  other than memory of the period (e.g. general sector priors or prompt-token effects), and the leakage
  reading of the historical gap is weakened.

Nothing is tuned: model id, prompt format, weekly cadence, seed and costs are pinned in `forward/config.json`
(same as the historical study). Each decision is appended to `forward/decisions.jsonl` (hash-chained) and committed
before the next US open; `git log --format='%H %cI' -- forward/decisions.jsonl` gives the commit timestamps.
The model's training cutoff is the vendor's to state; this repo does not assume one.
"""


def build(forward_dir: Path, out: Path) -> str:
    cfg = ForwardConfig.load_or_create(forward_dir)
    log = DecisionLog(forward_dir / "decisions.jsonl")
    n_rec = log.verify()
    recs = log.read()
    market = load_forward_market(Path(ROOT / "forward"))
    first = int(market.dates.searchsorted(pd.Timestamp(cfg.start)))
    lines = ["# Forward test (live paper trading)", "", HYPOTHESIS]
    lines += [f"Start: **{cfg.start}**; model `{cfg.model}` (provider {cfg.provider}), cadence {cfg.rebalance}, "
              f"T={cfg.temperature}, seed {cfg.seed}, costs {cfg.commission_bps:g}+{cfg.slippage_bps:g} bps/side, rf=0. "
              f"Hash chain verified over {n_rec} records.", ""]
    if first >= len(market) or not recs:
        lines += [f"**No forward data yet.** Newest stored bar: {market.dates[-1].date()}; decisions logged: {len(recs)}. "
                  "This page fills in as the daily workflow runs.", ""]
        out.write_text("\n".join(lines) + "\n")
        return "\n".join(lines)

    arms = {
        "LLM anon": replay_arm(ReplayStrategy(log, "anon", name="LLM anon"), market, cfg),
        "LLM real": replay_arm(ReplayStrategy(log, "real", name="LLM real"), market, cfg),
        "EW-15 (monthly rebal)": replay_arm(EqualWeight(TICKERS, rebalance=True), market, cfg, "monthly"),
        "SPY": replay_arm(SpyReference(), market, cfg, "monthly"),
    }
    served = sorted({r["model_served"] for r in recs})
    lines += [f"Bars: {arms['SPY'].equity.index[0].date()} .. {arms['SPY'].equity.index[-1].date()} "
              f"({len(arms['SPY'].equity)}); decisions: {sum(r['mode']=='anon' for r in recs)} anon / "
              f"{sum(r['mode']=='real' for r in recs)} real; model id actually served: {', '.join(served)}.", ""]
    lines += ["## Equity since start", "", format_table(summary_table(arms, rf=0.0)), ""]

    fig, ax = plt.subplots(figsize=(10, 5))
    for name, r in arms.items():
        ax.plot(r.equity.index, r.equity / r.equity.iloc[0], label=name, lw=2 if "LLM" in name else 1.2)
    ax.set_ylabel("equity (start = 1.0)")
    ax.set_title(f"Forward test since {cfg.start}")
    ax.legend()
    ax.grid(alpha=0.3)
    (ROOT / "reports" / "fig").mkdir(parents=True, exist_ok=True)
    fig.savefig(ROOT / "reports" / "fig" / "forward.png", dpi=130, bbox_inches="tight")
    lines += ["![forward equity](fig/forward.png)", ""]

    g = bootstrap_gap(gap_returns(arms["LLM real"].equity, arms["LLM anon"].equity))
    lines += ["## Anon vs real gap", ""]
    lines += [f"Mean daily return difference (real - anon), annualised: **{g['mean_annualised']:+.1%}** over {g['n_days']} days."]
    if g["ci"]:
        lo, hi = g["ci"]
        verdict = "includes zero (gap not distinguishable from nothing)" if lo <= 0 <= hi else "excludes zero (gap persists)"
        lines += [f"95% circular block bootstrap CI (block {g['block']}, 5000 resamples): [{lo:+.1%}, {hi:+.1%}], {verdict}."]
    else:
        lines += [f"95% CI withheld until n >= {MIN_DAYS_FOR_CI} trading days (n = {g['n_days']}); any reading now is noise."]
    lines += ["", "For scale, the historical 2022 gap was +10.3 pp over one year (real +11.9% vs anon +1.6%), same models/prompt.", ""]

    by = {m: {r["decision_date"]: r for r in recs if r["mode"] == m} for m in ("anon", "real")}
    common = sorted(set(by["anon"]) & set(by["real"]))
    if common:
        lines += ["## Decisions (exposure, number of names held)", "", "| date | anon exp | anon names | real exp | real names | decided at (UTC) |", "|---|---|---|---|---|---|"]
        for d in common[-20:]:
            a, r = by["anon"][d], by["real"][d]
            lines.append(f"| {d} | {a['exposure']:.2f} | {len(a['weights'])} | {r['exposure']:.2f} | {len(r['weights'])} | {r['decided_at_utc']} |")
        lines.append("")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--forward-dir", default=str(ROOT / "forward"))
    ap.add_argument("--out", default=str(ROOT / "reports" / "forward.md"))
    a = ap.parse_args()
    print(build(Path(a.forward_dir), Path(a.out)))


if __name__ == "__main__":
    main()
