"""Within-family replication check: the 2022 real-vs-anon comparison for several Gemini model ids, from the saved
runs (no model calls). Writes reports/within-family.md.

    .venv/bin/python scripts/within_family.py --models gemini-3.5-flash-lite gemini-3.1-flash-lite
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sab.engine import BacktestResult  # noqa: E402
from sab.metrics.metrics import compute_metrics  # noqa: E402

RUN = ROOT / "reports" / "runs" / "llm" / "2022_drawdown" / "with_costs_10_5_bps"
TERMS = {"energy/oil": r"energy|\boil\b", "tech": r"\btech", "healthcare": r"health", "defensive": r"defensive",
         "real ticker symbol": r"\b(AAPL|MSFT|NVDA|AMZN|GOOGL|JPM|BAC|XOM|CVX|JNJ|PFE|UNH|PG|KO|WMT)\b"}


def load(model: str, mode: str) -> BacktestResult | None:
    d = RUN / f"LLM_gemini_{model}_{mode}"
    return BacktestResult.load(d) if d.exists() else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--out", default=str(ROOT / "reports" / "within-family.md"))
    a = ap.parse_args()
    ew = BacktestResult.load(RUN / "EW-rebal_monthly")
    rows, extra = [], []
    for m in a.models:
        r, n = load(m, "real"), load(m, "anon")
        if r is None or n is None:
            extra.append(f"- `{m}`: runs missing (real={'ok' if r else 'no'}, anon={'ok' if n else 'no'})")
            continue
        mr, mn = compute_metrics(r), compute_metrics(n)
        wr, wn = r.weights.to_numpy().ravel(), n.weights.to_numpy().ravel()
        corr = float(np.corrcoef(wr, wn)[0, 1])
        cash_r = float((r.decisions["exposure"] < 0.05).mean())
        cash_n = float((n.decisions["exposure"] < 0.05).mean())
        txt = {k: [sum(bool(re.search(p, str(x), re.I)) for x in dec.decisions["rationale"]) for dec in (r, n)] for k, p in TERMS.items()}
        rows.append((m, mr, mn, corr, cash_r, cash_n, len(r.decisions), len(n.decisions), txt))
    L = ["# Within-family check: does the 2022 real-vs-anon gap replicate on another Gemini model id?", "",
         "Same engine, prompt, weekly cadence, T=0, seed 0, costs 10+5 bps, rf=0, period 2022; only the model id changes "
         "(**gemini-3.1-flash-lite** is a different release in the same flash-lite tier -- a within-family check, NOT cross-family).", "",
         "| model | arm | 2022 total return | CAGR | Sharpe | MaxDD | turnover/yr | mean exposure | decisions all-cash |", "|---|---|---|---|---|---|---|---|---|"]
    for m, mr, mn, corr, cr, cn, nr, nn, txt in rows:
        for lab, mm, c in (("real", mr, cr), ("anon", mn, cn)):
            L.append(f"| {m} | {lab} | {mm['TotalRet']:.1%} | {mm['CAGR']:.1%} | {mm['Sharpe']:.2f} | {mm['MaxDD']:.1%} | {mm['Turnover']:.1f} | {mm['Exposure']:.0%} | {c:.0%} |")
    e = compute_metrics(ew)
    L.append(f"| EW-15 rebal (benchmark) | - | {e['TotalRet']:.1%} | {e['CAGR']:.1%} | {e['Sharpe']:.2f} | {e['MaxDD']:.1%} | {e['Turnover']:.1f} | {e['Exposure']:.0%} | - |")
    L += ["", "## Gap (real minus anon)", "", "| model | return gap (pp) | Sharpe gap | weight-path correlation real/anon | decisions |", "|---|---|---|---|---|"]
    for m, mr, mn, corr, cr, cn, nr, nn, txt in rows:
        L.append(f"| {m} | {100 * (mr['TotalRet'] - mn['TotalRet']):+.1f} | {mr['Sharpe'] - mn['Sharpe']:+.2f} | {corr:.2f} | {nr}/{nn} |")
    L += ["", "## Words in the rationales that the prompt never supplied (count of decisions, real / anon)", "",
          "| term | " + " | ".join(r[0] for r in rows) + " |", "|---|" + "---|" * len(rows)]
    for k in TERMS:
        L.append(f"| {k} | " + " | ".join(f"{r[8][k][0]} / {r[8][k][1]}" for r in rows) + " |")
    L += [""] + extra
    Path(a.out).write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
