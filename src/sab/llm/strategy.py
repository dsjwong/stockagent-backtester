"""LLMAgent: a Strategy whose decision is a (cached) model completion.

Per decision bar:  ctx.prices(hist) -> build_prompt -> complete() [disk cache] -> parse
(retry once, then hold) -> target_orders.  The agent only ever reads `ctx.prices(...)`,
`ctx.weights()` and `ctx.portfolio.{cash,equity}`, so lookahead safety is inherited from the
engine's `iloc[:t+1]` view; the prompt-level audit (`assert_prompt_no_future`) and the
perturbation harness in tests/test_llm.py prove it end to end with the mock provider.

Every decision is logged (date, prompt hash, cache hit, weights, rationale, repairs, tokens)
and surfaced as `agent.decisions_frame()`, which `run_backtest` attaches to the result as
`result.decisions` and `BacktestResult.save` writes as decisions.parquet.

Defaults (decisions.md #22): mode=anon, lookback=60, weekly rebalance, temperature=0, seed=0.
`warmup` = 127 bars (6-month return window + 1) so the first prompt has every trailing return
populated; the runner passes it like it does for the classical strategies.
"""

from __future__ import annotations

from typing import Any, Iterable

import pandas as pd

from sab.engine.backtest import Context
from sab.engine.types import Order
from sab.llm.cache import DEFAULT_CACHE, LLMCache
from sab.llm.parse import REPAIR_KEYS, decide_with_retry
from sab.llm.prompt import RETURN_WINDOWS, AnonMapping, Mode, PromptSpec, build_prompt
from sab.llm.providers import LLMResponse, complete
from sab.strategies._base import target_orders, universe

HIST_BARS = RETURN_WINDOWS["6m"] + 1
_UNSET: Any = object()


class LLMAgent:
    def __init__(self, provider: str = "mock", model: str = "mock", mode: Mode = "anon", lookback: int = 60,
                 rebalance: str = "weekly", temperature: float = 0.0, seed: int = 0, *,
                 tickers: Iterable[str] | None = None, max_tokens: int = 400, cache: LLMCache | None = DEFAULT_CACHE,
                 bypass_cache: bool = False, max_retries: int = 1, verbose: bool = False) -> None:
        if mode not in ("real", "anon"):
            raise ValueError("mode must be 'real' or 'anon'")
        if rebalance not in ("daily", "weekly", "monthly"):
            raise ValueError("rebalance must be daily/weekly/monthly")
        self.provider, self.model, self.mode = provider, model, mode
        self.lookback, self.rebalance = int(lookback), rebalance
        self.temperature, self.seed = float(temperature), int(seed)
        self.tickers = universe(tickers)
        self.max_tokens, self.cache, self.bypass_cache, self.max_retries = int(max_tokens), cache, bypass_cache, int(max_retries)
        self.mapping = AnonMapping.make(self.tickers, seed=self.seed) if mode == "anon" else AnonMapping.identity(self.tickers)
        self.warmup = max(self.lookback, HIST_BARS)
        self.name = f"LLM({provider}/{model}, {mode})"
        self._initial_equity: float | None = None
        self._log: list[dict[str, Any]] = []
        self.n_provider_calls = 0            # completions that were NOT served from cache
        self.verbose = bool(verbose)         # one progress line per decision (real-provider runs take minutes)

    # ------------------------------------------------------------------ bookkeeping
    @property
    def params(self) -> dict[str, Any]:
        return {"provider": self.provider, "model": self.model, "mode": self.mode, "lookback": self.lookback,
                "temperature": self.temperature, "seed": self.seed, "max_tokens": self.max_tokens,
                "rebalance": self.rebalance, "warmup": self.warmup, "max_retries": self.max_retries,
                "anon_mapping": self.mapping.to_dict() if self.mode == "anon" else None}

    def decisions_frame(self) -> pd.DataFrame:
        cols = ["date", "bar_index", "prompt_sha256", "cache_hit", "n_calls", "prompt_tokens", "completion_tokens",
                "latency_s", "exposure", "n_held", "weights", "rationale", "retried", "fallback", "n_repairs", "repairs", "error"]
        if not self._log:
            return pd.DataFrame(columns=cols)
        df = pd.DataFrame(self._log)[cols]
        return df.set_index("date")

    @property
    def decisions(self) -> pd.DataFrame:
        return self.decisions_frame()

    # ------------------------------------------------------------------ pieces
    def portfolio_state(self, ctx: Context) -> dict[str, Any]:
        if self._initial_equity is None:
            self._initial_equity = float(ctx.portfolio.equity)
        return {"weights": ctx.weights(), "cash": float(ctx.portfolio.cash), "equity": float(ctx.portfolio.equity),
                "initial_equity": self._initial_equity}

    def prompt_for(self, ctx: Context) -> PromptSpec:
        px = ctx.prices(max(self.lookback, HIST_BARS))[self.tickers]
        return build_prompt(px, self.portfolio_state(ctx), self.mode, self.lookback, mapping=self.mapping,
                            day_index=ctx.bar_index)

    def ask(self, prompt: str, *, temperature: float | None = None, seed: int | None = _UNSET,
            bypass_cache: bool | None = None, cache: LLMCache | None = None, use_cache: bool = True) -> LLMResponse:
        """One completion with the agent's settings; overrides are for consistency_check."""
        resp = complete(self.provider, self.model, prompt, temperature=self.temperature if temperature is None else temperature,
                        max_tokens=self.max_tokens, seed=self.seed if seed is _UNSET else seed,
                        cache=(self.cache if cache is None else cache) if use_cache else None,
                        bypass_cache=self.bypass_cache if bypass_cache is None else bypass_cache)
        if not resp.cached:
            self.n_provider_calls += 1
        return resp

    def decide(self, ctx: Context) -> tuple[dict[str, float], dict[str, Any]]:
        """Weights over real tickers + meta (prompt spec, model meta, log row)."""
        spec = self.prompt_for(ctx)
        prev = {spec.mapping.to_anon[t]: w for t, w in ctx.weights().items() if t in spec.mapping.to_anon}
        w_names, meta = decide_with_retry(self.ask, spec.text, spec.names, prev_weights=prev, max_retries=self.max_retries)
        weights = {spec.mapping.to_real[n]: w for n, w in w_names.items()}
        responses: list[LLMResponse] = meta["responses"]
        row = {
            "date": ctx.date, "bar_index": ctx.bar_index, "prompt_sha256": spec.sha256,
            "cache_hit": all(r.cached for r in responses), "n_calls": len(responses),
            "prompt_tokens": sum(r.usage.get("prompt_tokens", 0) for r in responses),
            "completion_tokens": sum(r.usage.get("completion_tokens", 0) for r in responses),
            "latency_s": sum(r.latency_s for r in responses),
            "exposure": float(sum(weights.values())), "n_held": sum(1 for v in weights.values() if v > 1e-6),
            "weights": {k: round(v, 6) for k, v in weights.items() if v > 1e-6},
            "rationale": meta["rationale"], "retried": meta["retried"], "fallback": meta["fallback"],
            "n_repairs": sum(meta["repairs"].get(k, 0) for k in REPAIR_KEYS), "repairs": {k: v for k, v in meta["repairs"].items() if v},
            "error": meta["error"],
        }
        self._log.append(row)
        if self.verbose:
            print(f"  [{len(self._log):3d}] {pd.Timestamp(ctx.date).date()} {'cache' if row['cache_hit'] else 'CALL '} "
                  f"calls={row['n_calls']} exp={row['exposure']:.2f} held={row['n_held']:2d} "
                  f"{'RETRY ' if row['retried'] else ''}{'FALLBACK ' if row['fallback'] else ''}"
                  f"rep={row['n_repairs']} tok={row['prompt_tokens']}+{row['completion_tokens']} {row['latency_s']:.1f}s", flush=True)
        return weights, {"spec": spec, **meta, "row": row}

    def on_bar(self, ctx: Context) -> list[Order]:
        weights, _ = self.decide(ctx)
        return target_orders(ctx, weights, self.tickers)


# ---------------------------------------------------------------------- consistency (M4b helper)

def consistency_check(agent: LLMAgent, ctx: Context, n: int = 3, temperature: float = 0.7,
                      decimals: int = 4) -> dict[str, Any]:
    """Ask the SAME prompt `n` times at `temperature` (cache bypassed, nothing written) and
    report how often the parsed weights agree. `fraction_identical` = share of the n answers
    equal to the most common answer (1.0 = perfectly consistent). For a real provider `seed`
    is left None so the model is free to vary; mock_random draws a fresh seed per call, mock and
    mock_ew are pure functions of the prompt (fraction 1.0)."""
    from collections import Counter

    from sab.llm.parse import ParseError, parse_weights

    spec = agent.prompt_for(ctx)
    answers, texts = [], []
    for _ in range(int(n)):
        resp = agent.ask(spec.text, temperature=temperature, seed=None, use_cache=False)
        texts.append(resp.text)
        try:
            w = parse_weights(resp.text, spec.names).weights
            answers.append(tuple(sorted((k, round(v, decimals)) for k, v in w.items())))
        except ParseError:
            answers.append(("<unparseable>",))
    counts = Counter(answers)
    top = counts.most_common(1)[0][1] if counts else 0
    return {"n": int(n), "temperature": float(temperature), "n_distinct": len(counts),
            "fraction_identical": top / n if n else float("nan"), "prompt_sha256": spec.sha256,
            "answers": [dict(a) if a != ("<unparseable>",) else None for a in answers], "texts": texts}
