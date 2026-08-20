"""Figures + tables for docs/leakage-writeup.md -- the real-vs-anonymised (memorisation) story.

Reads only the SAVED runs under reports/runs/ (BacktestResult.load) and the consistency JSONs
under reports/runs/llm/consistency/. No model calls, no network. The one thing it re-runs is the
zero-cost *replay* of the LLM's logged weights (report_llm.py::replay_no_cost, decisions.md #29),
which is pure engine work and needs no key.

    .venv/bin/python scripts/fig_leakage.py

Writes:
    reports/fig/leakage_2022.png          (a) 2022 equity: real vs anon vs EW-rebal
    reports/fig/leakage_consistency.png   (b) 8 repeated draws per mode from the same prompt
    reports/fig/leakage_words.png         (c) sector/regime words in the rationales, real vs anon
    reports/fig/leakage_bottomline.png    (d) full-period anon vs EW-rebal vs EW-hold, with costs

and prints the markdown tables (c) and (d) so the numbers in the writeup stay derived, not typed.
"""

from __future__ import annotations

import json
import re
import sys
import textwrap
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from report_llm import replay_no_cost  # noqa: E402
from sab.engine import BacktestResult, load_market  # noqa: E402
from sab.metrics.metrics import compute_metrics  # noqa: E402

REPORTS = ROOT / "reports"
FIG = REPORTS / "fig"
RUNS = REPORTS / "runs"
COST_DIR = "with_costs_10_5_bps"
AGENT = "LLM_gemini_gemini-3.5-flash-lite_{mode}"

C_REAL, C_ANON, C_EW, C_EWH, C_CASH = "#ff7f0e", "#d62728", "#2ca02c", "#8c564b", "#cccccc"

# Concepts counted in the rationales. Sector/regime vocabulary is not in the prompt: the prompt
# has prices and trailing returns only (decisions.md #23), and in anon mode not even a ticker.
CONCEPTS: list[tuple[str, str]] = [
    ("energy / oil", r"\benerg|\boil\b"),
    ("tech / technology", r"\btech"),
    ("healthcare", r"\bhealthcare\b"),
    ("consumer staples", r"\bstaples\b"),
    ("financials", r"\bfinancial"),
    ('"defensive"', r"\bdefensiv"),
    ("high-beta / cyclicals", r"high-beta|\bcyclical"),
    ("drawdown / correction", r"\bdrawdown|\bcorrection\b"),
    ("a real ticker symbol", r"\b(AAPL|MSFT|NVDA|AMZN|GOOGL|JPM|BAC|XOM|CVX|JNJ|PFE|UNH|PG|KO|WMT)\b"),
]


def agent_dir(period_slug: str, mode: str) -> Path:
    return RUNS / "llm" / period_slug / COST_DIR / AGENT.format(mode=mode)


def load(period_slug: str, name: str, llm: bool = True) -> BacktestResult:
    base = RUNS / "llm" / period_slug / COST_DIR if llm else RUNS / period_slug / COST_DIR
    return BacktestResult.load(base / name)


def md_table(rows: list[list[str]], header: list[str]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


# --------------------------------------------------------------------- (a) 2022 equity curves

def fig_2022() -> None:
    real = load("2022_drawdown", AGENT.format(mode="real"))
    anon = load("2022_drawdown", AGENT.format(mode="anon"))
    ew = load("2022_drawdown", "EW-rebal_monthly")
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    for res, color, label, lw in ((real, C_REAL, "LLM, REAL tickers + dates", 2.4),
                                  (anon, C_ANON, "LLM, ANONYMISED (Stock 01..15, Day t)", 2.4),
                                  (ew, C_EW, "EW-15 rebal (benchmark)", 1.6)):
        eq = res.equity / res.equity.iloc[0]
        ax.plot(eq.index, eq, color=color, lw=lw, label=label, zorder=5 if lw > 2 else 3)
        ax.annotate(f"{eq.iloc[-1] - 1:+.1%}", xy=(eq.index[-1], eq.iloc[-1]), xytext=(6, -3),
                    textcoords="offset points", color=color, fontsize=10, fontweight="bold")
    ax.axhline(1.0, color="#666666", lw=0.8, ls=":")
    ax.set_title("2022 drawdown: same engine, same prices, two prompt modes\n"
                 "(growth of $1, costs 10+5 bps/side)", fontsize=12)
    ax.set_ylabel("growth of $1")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="lower left")
    ax.margins(x=0.06)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIG / "leakage_2022.png", dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------- (b) consistency panel

def consistency_draws(mode: str) -> tuple[list[dict[str, float]], list[str], list[str]]:
    """8 draws (5 at T=0.7 then 3 at T=0) as {ticker: weight}, their labels and first rationales."""
    d = RUNS / "llm" / "consistency"
    weights, labels, quotes = [], [], []
    for suffix, temp in (("T0.7_n5", "T=0.7"), ("T0_n3", "T=0")):
        j = json.loads((d / f"{AGENT.format(mode=mode)}_2022-06-13_{suffix}.json").read_text())
        real_answers = j.get("answers_real") or []
        for i, ans in enumerate(j["answers"]):
            w = real_answers[i] if (i < len(real_answers) and real_answers[i]) else ans
            weights.append({k: float(v) for k, v in w.items() if float(v) > 0})
            labels.append(f"{temp}\n#{i + 1}")
        quotes.append(json.loads(j["texts"][0])["rationale"])
    return weights, labels, quotes


def fig_consistency() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.0), sharey=True)
    captions: list[str] = []
    palette = {"XOM": "#8c564b", "CVX": "#e377c2", "JNJ": "#1f77b4", "KO": "#9467bd"}
    for ax, mode, title in ((axes[0], "real", "REAL tickers + dates"), (axes[1], "anon", "ANONYMISED")):
        weights, labels, quotes = consistency_draws(mode)
        for x, w in enumerate(weights):
            bottom = 0.0
            for tkr in sorted(w, key=lambda t: -w[t]):
                ax.bar(x, w[tkr], bottom=bottom, color=palette.get(tkr, "#7f7f7f"), edgecolor="white", lw=0.6)
                if w[tkr] >= 0.09:
                    ax.text(x, bottom + w[tkr] / 2, tkr, ha="center", va="center", fontsize=7, color="white")
                bottom += w[tkr]
            if bottom < 0.999:                       # the rest is cash
                ax.bar(x, 1.0 - bottom, bottom=bottom, color=C_CASH, edgecolor="white", lw=0.6)
                if 1.0 - bottom >= 0.3:
                    ax.text(x, bottom + (1 - bottom) / 2, "CASH", ha="center", va="center", fontsize=8, color="#444444")
        invested = sum(1 for w in weights if sum(w.values()) > 0)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylim(0, 1.02)
        ax.set_title(f"{title}\ninvested in {invested}/8 draws", fontsize=11)
        ax.grid(axis="y", alpha=0.3)
        captions.append("\n".join(textwrap.wrap(f'first rationale: "{quotes[0]}"', 64)))
    axes[0].set_ylabel("target weight")
    fig.suptitle("Same prompt, same all-cash state, 2022-06-13, 8 fresh calls per mode (cache bypassed)", fontsize=12)
    fig.tight_layout(rect=(0, 0.19, 1, 0.96))
    for ax, caption in zip(axes, captions):
        box = ax.get_position()
        fig.text(box.x0 + box.width / 2, 0.15, caption, ha="center", va="top", fontsize=8.5,
                 style="italic", color="#333333")
    fig.savefig(FIG / "leakage_consistency.png", dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------- (c) rationale vocabulary

def rationales(mode: str) -> list[str]:
    out: list[str] = []
    for period in ("2021_bull", "2022_drawdown"):
        res = load(period, AGENT.format(mode=mode))
        out += [str(s) for s in res.decisions["rationale"].dropna()]
    return out


def fig_words() -> str:
    texts = {m: rationales(m) for m in ("real", "anon")}
    n = {m: len(t) for m, t in texts.items()}
    counts = {m: [sum(1 for s in texts[m] if re.search(pat, s, re.I)) for _, pat in CONCEPTS]
              for m in ("real", "anon")}
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    y = range(len(CONCEPTS))
    ax.barh([i + 0.2 for i in y], counts["real"], height=0.38, color=C_REAL, label=f"REAL mode ({n['real']} rationales)")
    ax.barh([i - 0.2 for i in y], counts["anon"], height=0.38, color=C_ANON, label=f"ANON mode ({n['anon']} rationales)")
    for i in y:
        ax.text(counts["real"][i] + 1, i + 0.2, str(counts["real"][i]), va="center", fontsize=8, color=C_REAL)
        ax.text(counts["anon"][i] + 1, i - 0.2, str(counts["anon"][i]), va="center", fontsize=8, color=C_ANON)
    ax.set_yticks(list(y))
    ax.set_yticklabels([label for label, _ in CONCEPTS], fontsize=9)
    ax.set_xlabel("rationales containing the term (2021 + 2022, 104 decisions per mode)")
    ax.set_title("Vocabulary the prompt never supplied\n"
                 "sector and regime language appears almost only when the tickers are named", fontsize=12)
    ax.grid(axis="x", alpha=0.3)
    ax.set_xlim(0, max(counts["real"]) * 1.12)
    ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG / "leakage_words.png", dpi=130)
    plt.close(fig)
    rows = [[label, str(counts["real"][i]), str(counts["anon"][i])] for i, (label, _) in enumerate(CONCEPTS)]
    return md_table(rows, ["term in the rationale", f"REAL ({n['real']} decisions)", f"ANON ({n['anon']} decisions)"])


# --------------------------------------------------------------------- (d) honest bottom line

def bottom_line() -> str:
    market = load_market()
    period, label = "full_2019-01_.._2026-08", "full (2019-01 .. 2026-08)"
    anon = load(period, AGENT.format(mode="anon"))
    rows_src = [("LLM, anonymised (weekly)", anon, replay_no_cost(anon, market, label)),
                ("EW-15 rebal (benchmark, monthly)", load(period, "EW-rebal_monthly"),
                 BacktestResult.load(RUNS / period / "no_costs" / "EW-rebal_monthly")),
                ("EW-15 buy & hold", load(period, "EW-hold"),
                 BacktestResult.load(RUNS / period / "no_costs" / "EW-hold"))]
    rows, bars = [], []
    for name, res, free in rows_src:
        m, mf = compute_metrics(res), compute_metrics(free)
        drag = mf["CAGR"] - m["CAGR"]
        rows.append([name, f"{m['CAGR']:.1%}", f"{m['Sharpe']:.2f}", f"{m['MaxDD']:.1%}",
                     f"{m['Turnover']:.1f}x", f"${m['Costs']:,.0f}", f"{drag * 100:.1f} pp/yr"])
        bars.append((name, m["CAGR"], mf["CAGR"], m["Sharpe"]))

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.6))
    x = range(len(bars))
    colors = [C_ANON, C_EW, C_EWH]
    ax.bar(x, [b[2] for b in bars], color=colors, alpha=0.28, label="before costs")
    ax.bar(x, [b[1] for b in bars], color=colors, label="after costs")
    for i, b in enumerate(bars):
        ax.text(i, b[2] + 0.008, f"{b[1]:.1%}", ha="center", fontsize=9, fontweight="bold")
    ax.text(0, (bars[0][1] + bars[0][2]) / 2, f"-{(bars[0][2] - bars[0][1]) * 100:.1f} pp costs",
            ha="center", va="center", fontsize=8, color="#8b1a1a")
    ax.set_title("CAGR, 2019-2026 (with costs, faded = before costs)", fontsize=11)
    ax.set_ylabel("CAGR")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1))
    ax2.bar(x, [b[3] for b in bars], color=colors)
    for i, b in enumerate(bars):
        ax2.text(i, b[3] + 0.02, f"{b[3]:.2f}", ha="center", fontsize=9, fontweight="bold")
    ax2.set_title("Sharpe (rf = 0), 2019-2026", fontsize=11)
    for a in (ax, ax2):
        a.set_xticks(list(x))
        a.set_xticklabels(["LLM\n(anon)", "EW-15\nrebal", "EW-15\nhold"], fontsize=9)
        a.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")
    fig.suptitle("After the controls, the LLM does not beat equal weight risk-adjusted", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG / "leakage_bottomline.png", dpi=130)
    plt.close(fig)
    return md_table(rows, ["strategy (2019-2026, costs 10+5 bps/side, rf=0)", "CAGR", "Sharpe", "MaxDD",
                           "turnover/yr", "costs paid", "cost drag"])


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig_2022()
    fig_consistency()
    words = fig_words()
    bottom = bottom_line()
    print("\n### (c) rationale vocabulary\n")
    print(words)
    print("\n### (d) bottom line\n")
    print(bottom)
    print("\nwrote:", ", ".join(sorted(p.name for p in FIG.glob("leakage_*.png"))))


if __name__ == "__main__":
    main()
