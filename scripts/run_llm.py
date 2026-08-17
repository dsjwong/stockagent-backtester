"""M4: run an LLM agent (mock or real provider) against the benchmark set on one period, or
estimate the token/cost budget of a run without calling any model. Offline unless a real
provider is chosen. Examples:

    .venv/bin/python scripts/run_llm.py --provider mock --model mock_ew --mode anon --period 2022
    .venv/bin/python scripts/run_llm.py --estimate-tokens --period 2021 --period 2022 --period full
    .venv/bin/python scripts/run_llm.py --provider anthropic --model claude-haiku-4-5 --mode anon --period 2022

Comparison set (decisions.md #22): EW-rebal monthly (official), EW-hold, SPY (reference), both
Trend sizings (weekly), WeightedTrend (monthly). Period handling is the same as run_classical.py
(entered from cash on the period's first bar with pre-period warmup history). Results go under
reports/runs/llm/<period>/<agent-slug>/ (equity/weights/trades/decisions parquet + config.json)
and reports/runs/llm/<period>/<agent-slug>.md (metrics table).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_classical import PERIODS, run_period, slug  # noqa: E402
from sab.data.universe import BENCHMARK, TICKERS  # noqa: E402
from sab.engine import CostModel, ZeroCost, load_market  # noqa: E402
from sab.engine.backtest import rebalance_bars  # noqa: E402
from sab.engine.types import PortfolioState  # noqa: E402
from sab.llm import LLMAgent, build_prompt  # noqa: E402
from sab.llm.providers import approx_tokens  # noqa: E402
from sab.llm.strategy import HIST_BARS  # noqa: E402
from sab.metrics.metrics import format_table, summary_table  # noqa: E402
from sab.strategies import EqualWeight, SpyReference, TrendFollowing, WeightedTrend  # noqa: E402

RUNS = ROOT / "reports" / "runs" / "llm"
PERIOD_ALIASES = {"full": "full (2019-01 .. 2026-08)", "2021": "2021 (bull)", "2022": "2022 (drawdown)",
                  "2023": "2023-01 .. 2026-08 (recovery)"}

# USD per 1M tokens (input, output). Prices as of 2026-06-24 (Anthropic model docs cache; Sonnet 5 lists a
# $2/$10 intro rate through 2026-08-31 -- the sticker price is used here). VERIFY on the provider's pricing
# page before running anything real; hosted prices change and other providers are not listed because no
# verified figure was available when this was written.
PRICES_USD_PER_MTOK: dict[str, tuple[float, float]] = {
    "anthropic/claude-haiku-4-5": (1.00, 5.00),
    "anthropic/claude-sonnet-5": (3.00, 15.00),
    "anthropic/claude-opus-5": (5.00, 25.00),
    "ollama/<local model>": (0.0, 0.0),
}
PRICES_AS_OF = "2026-06-24"
COMPLETION_TOKENS_PER_CALL = 120       # ~15 weights + rationale; mocks produce ~90


def resolve_period(p: str) -> tuple[str, str | None, str | None]:
    key = PERIOD_ALIASES.get(p, p)
    if key in PERIODS:
        return key, *PERIODS[key]
    if ":" in p:                                   # YYYY-MM-DD:YYYY-MM-DD
        a, b = p.split(":", 1)
        return p, a or None, b or None
    raise SystemExit(f"unknown period {p!r}; choose from {list(PERIOD_ALIASES)} or START:END")


def benchmarks() -> list[tuple[str, object, str, int]]:
    return [
        ("EW-rebal (monthly)", EqualWeight(TICKERS, rebalance=True), "monthly", 0),
        ("EW-hold", EqualWeight(TICKERS, rebalance=False), "monthly", 0),
        ("SPY (reference)", SpyReference(), "monthly", 0),
        ("Trend(SMA50>SMA200)", TrendFollowing(), "weekly", 200),
        ("Trend(SMA50>SMA200), 1/N slots", TrendFollowing(sizing="slots"), "weekly", 200),
        ("WeightedTrend(21/63/126/252)", WeightedTrend(), "monthly", 252),
    ]


def run(args) -> None:
    market = load_market(list(TICKERS) + [BENCHMARK])
    cost = ZeroCost if args.no_costs else CostModel(10.0, 5.0)
    for p in args.period:
        label, start, end = resolve_period(p)
        agent = LLMAgent(args.provider, args.model, args.mode, lookback=args.lookback, rebalance=args.rebalance,
                         temperature=args.temperature, seed=args.seed, bypass_cache=args.no_cache)
        res = run_period(agent, market, start, end, agent.rebalance, agent.warmup, cost)
        res.decisions = agent.decisions_frame()          # trimmed result keeps the full log (dates are within period anyway)
        runs = {agent.name: res}
        for name, strat, reb, warm in benchmarks():
            runs[name] = run_period(strat, market, start, end, reb, warm, cost)
        table = format_table(summary_table(runs, rf=0.0))
        dec = res.decisions
        n_calls = int(dec["n_calls"].sum()) if len(dec) else 0
        summary = {
            "period": label, "costs": cost.describe(), "agent": agent.params, "n_decisions": int(len(dec)),
            "n_model_calls": n_calls, "n_provider_calls_this_run": agent.n_provider_calls,
            "cache_hits": int(dec["cache_hit"].sum()) if len(dec) else 0,
            "prompt_tokens": int(dec["prompt_tokens"].sum()) if len(dec) else 0,
            "completion_tokens": int(dec["completion_tokens"].sum()) if len(dec) else 0,
            "retries": int(dec["retried"].sum()) if len(dec) else 0, "fallbacks": int(dec["fallback"].sum()) if len(dec) else 0,
            "repairs": int(dec["n_repairs"].sum()) if len(dec) else 0,
            "mean_exposure": float(dec["exposure"].mean()) if len(dec) else float("nan"),
        }
        print(f"\n== {label} | {agent.name} | {'no costs' if args.no_costs else 'costs 10+5 bps'} ==")
        print(table)
        print(json.dumps({k: v for k, v in summary.items() if k != "agent"}, indent=1, default=str))
        if not args.no_save:
            out = RUNS / slug(label) / ("no_costs" if args.no_costs else "with_costs_10_5_bps")
            out.mkdir(parents=True, exist_ok=True)
            d = res.save(out, name=slug(agent.name))
            for name, r in runs.items():
                if name != agent.name:
                    r.save(out, name=slug(name))
            (d / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
            (out / f"{slug(agent.name)}.md").write_text(
                f"# {agent.name} on {label}\n\n{'no costs' if args.no_costs else 'costs 10+5 bps/side'}, rf=0\n\n{table}\n\n"
                f"```json\n{json.dumps(summary, indent=1, default=str)}\n```\n")
            print(f"saved under {out}")


def estimate(args) -> None:
    """Build every prompt the agent would see on each period (all-cash portfolio state; the length
    barely depends on holdings) and price it. No model is called, nothing is cached."""
    market = load_market(list(TICKERS))
    rows = []
    for p in args.period:
        label, start, end = resolve_period(p)
        dates = market.dates
        first = int(dates.searchsorted(pd.Timestamp(start))) if start else 0
        warm = max(args.lookback, HIST_BARS)
        lo = max(first - warm, 0)
        m = market.slice(dates[lo], pd.Timestamp(end) if end else None)
        agent = LLMAgent("mock", "mock", args.mode, lookback=args.lookback, rebalance=args.rebalance, seed=args.seed, cache=None)
        bars = sorted(rebalance_bars(m.dates, args.rebalance, first - lo))
        st = PortfolioState(cash=1_000_000.0)
        toks = []
        for t in bars:
            st.mark(m.dates[t], m.close.iloc[t])
            px = m.close.iloc[: t + 1][list(TICKERS)].iloc[-warm:]
            spec = build_prompt(px, {"weights": {}, "cash": st.cash, "equity": st.equity, "initial_equity": 1e6},
                                args.mode, args.lookback, mapping=agent.mapping, day_index=t)
            toks.append(approx_tokens(spec.text))
        rows.append({"period": label, "rebalance": args.rebalance, "calls": len(bars), "prompt_tokens": sum(toks),
                     "avg_prompt_tokens": sum(toks) / max(len(toks), 1),
                     "completion_tokens": len(bars) * COMPLETION_TOKENS_PER_CALL})
    df = pd.DataFrame(rows).set_index("period")
    print(f"\nToken estimate (mode={args.mode}, lookback={args.lookback}, ~4 chars/token, {COMPLETION_TOKENS_PER_CALL} completion tokens/call, no retries):")
    print(df.round(0).to_string())
    print(f"\nCost table, USD per full run of the period. Prices as of {PRICES_AS_OF} -- VERIFY on the provider pricing page before running.")
    cost = pd.DataFrame({m: [(r["prompt_tokens"] * i + r["completion_tokens"] * o) / 1e6 for r in rows]
                         for m, (i, o) in PRICES_USD_PER_MTOK.items()}, index=[r["period"] for r in rows])
    print(cost.round(3).to_string())
    print("\nNotes: one call per rebalance bar; a parse retry doubles that bar's cost; a consistency check (n=3) triples it; "
          "with thinking-on models the completion side can be several times larger than the estimate.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--provider", default="mock")
    ap.add_argument("--model", default="mock")
    ap.add_argument("--mode", default="anon", choices=["real", "anon"])
    ap.add_argument("--period", action="append", default=None, help="full | 2021 | 2022 | 2023 | START:END (repeatable)")
    ap.add_argument("--lookback", type=int, default=60)
    ap.add_argument("--rebalance", default="weekly", choices=["daily", "weekly", "monthly"])
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-costs", action="store_true")
    ap.add_argument("--no-cache", action="store_true", help="bypass the disk cache (still writes)")
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--estimate-tokens", action="store_true", help="only build prompts and print token/cost estimates")
    args = ap.parse_args()
    args.period = args.period or ["2022"]
    if args.estimate_tokens:
        estimate(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
