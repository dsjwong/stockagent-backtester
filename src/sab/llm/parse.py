"""Turn a model reply into valid target weights, or fail loudly and fall back.

Layers
1. `extract_json(text)`  -- tolerant JSON extraction: strips ``` fences, takes the first
   balanced {...} object, repairs trailing commas. Raises ParseError.
2. `validate_weights(obj, allowed)` -- keeps only `allowed` names (unknown dropped), non-numeric
   dropped, negatives clipped to 0, sum > 1 renormalised to 1, missing names -> 0. Every repair
   is COUNTED in `repairs` so a run can report how often the model broke the spec. Raises
   ParseError when there is no usable "weights" object at all.
3. `parse_weights(text, allowed)` -- 1 + 2 into a ParseResult.
4. `decide_with_retry(ask, prompt, allowed, prev_weights)` -- the StockAgent "Secretary" pattern:
   call the model, parse; on failure call ONCE more with the error appended to the prompt; if
   that fails too, hold `prev_weights` (fallback) and say so in the meta counts.

Names are matched exactly after whitespace-trim and case-fold ("aapl" -> "AAPL",
"stock 03" -> "Stock 03"); no fuzzy matching, so the log tells the truth about the model.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping

REPAIR_KEYS = ("unknown_dropped", "non_numeric_dropped", "negative_clipped", "renormalised", "missing_zeroed",
               "trailing_comma_fixed", "fence_stripped", "rationale_missing", "rationale_truncated")
MAX_RATIONALE_WORDS = 40


class ParseError(ValueError):
    pass


@dataclass
class ParseResult:
    weights: dict[str, float]                       # over ALL allowed names, in prompt namespace
    rationale: str = ""
    repairs: dict[str, int] = field(default_factory=lambda: {k: 0 for k in REPAIR_KEYS})
    raw: str = ""

    @property
    def n_repairs(self) -> int:
        return sum(self.repairs.values())


# --------------------------------------------------------------------------- extraction

_FENCE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)
_TRAILING_COMMA = re.compile(r",\s*([}\]])")


def _balanced_objects(text: str) -> Iterable[str]:
    """Yield every top-level {...} substring in order (brace-balanced, string-aware)."""
    depth, start, in_str, esc = 0, -1, False, False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0:
                yield text[start:i + 1]


def extract_json(text: str, repairs: dict[str, int] | None = None) -> dict[str, Any]:
    """Return the first JSON object found in `text` (fences and prose around it are ignored,
    trailing commas repaired). Raises ParseError if none parses."""
    repairs = repairs if repairs is not None else {}
    if not isinstance(text, str) or not text.strip():
        raise ParseError("empty response")
    body = text
    m = _FENCE.search(text)
    if m:
        body = m.group(1)
        repairs["fence_stripped"] = repairs.get("fence_stripped", 0) + 1
    candidates = list(_balanced_objects(body)) or list(_balanced_objects(text))
    if not candidates:
        raise ParseError("no JSON object ({...}) in response")
    last_err: Exception | None = None
    for cand in candidates:
        try:
            obj = json.loads(cand)
        except json.JSONDecodeError as e:
            fixed = _TRAILING_COMMA.sub(r"\1", cand)
            if fixed == cand:
                last_err = e
                continue
            try:
                obj = json.loads(fixed)
                repairs["trailing_comma_fixed"] = repairs.get("trailing_comma_fixed", 0) + 1
            except json.JSONDecodeError as e2:
                last_err = e2
                continue
        if isinstance(obj, dict):
            return obj
    raise ParseError(f"JSON object did not parse: {last_err}")


# --------------------------------------------------------------------------- validation

def _norm(name: str) -> str:
    return " ".join(str(name).split()).casefold()


def validate_weights(obj: Mapping[str, Any], allowed: list[str], repairs: dict[str, int] | None = None,
                     tol: float = 1e-9) -> tuple[dict[str, float], str]:
    """Return ({allowed name: weight}, rationale). See module docstring for the repair rules."""
    repairs = repairs if repairs is not None else {}

    def bump(k: str, n: int = 1) -> None:
        repairs[k] = repairs.get(k, 0) + n

    raw = obj.get("weights") if isinstance(obj, Mapping) else None
    if raw is None and isinstance(obj, Mapping) and all(isinstance(v, (int, float)) for v in obj.values()) and obj:
        raw = obj                                   # bare {"AAPL": 0.1, ...}
    if not isinstance(raw, Mapping):
        raise ParseError('missing "weights" object')
    lookup = {_norm(a): a for a in allowed}
    weights: dict[str, float] = {}
    for k, v in raw.items():
        name = lookup.get(_norm(k))
        if name is None:
            bump("unknown_dropped")
            continue
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            try:
                v = float(str(v).strip().rstrip("%"))
            except (TypeError, ValueError):
                bump("non_numeric_dropped")
                continue
        v = float(v)
        if v != v:                                  # NaN
            bump("non_numeric_dropped")
            continue
        if v < 0:
            bump("negative_clipped")
            v = 0.0
        weights[name] = weights.get(name, 0.0) + v  # duplicate keys after normalisation add up
    if not weights and raw:
        raise ParseError(f'no known tickers in "weights" (got {list(raw)[:5]})')
    total = sum(weights.values())
    if total > 1.0 + tol:
        bump("renormalised")
        weights = {k: v / total for k, v in weights.items()}
    elif total > 1.0:                               # float noise: snap to exactly 1, not a "repair"
        weights = {k: v / total for k, v in weights.items()}
    missing = [a for a in allowed if a not in weights]
    if missing:
        bump("missing_zeroed", len(missing))
    out = {a: float(weights.get(a, 0.0)) for a in allowed}

    rationale = obj.get("rationale", "") if isinstance(obj, Mapping) else ""
    if not isinstance(rationale, str) or not rationale.strip():
        bump("rationale_missing")
        rationale = ""
    else:
        words = rationale.split()
        if len(words) > MAX_RATIONALE_WORDS:
            bump("rationale_truncated")
            rationale = " ".join(words[:MAX_RATIONALE_WORDS]) + " ..."
    return out, rationale


def parse_weights(text: str, allowed: list[str]) -> ParseResult:
    repairs = {k: 0 for k in REPAIR_KEYS}
    obj = extract_json(text, repairs)
    weights, rationale = validate_weights(obj, allowed, repairs)
    return ParseResult(weights=weights, rationale=rationale, repairs=repairs, raw=text)


# --------------------------------------------------------------------------- retry / fallback

RETRY_SUFFIX = ("\n\nYour previous reply could not be used: {error}\n"
                "Reply again with ONLY the JSON object {{\"weights\": {{...}}, \"rationale\": \"...\"}} and nothing else.")


def decide_with_retry(ask: Callable[[str], Any], prompt: str, allowed: list[str],
                      prev_weights: Mapping[str, float] | None = None, max_retries: int = 1) -> tuple[dict[str, float], dict[str, Any]]:
    """`ask(prompt) -> object with .text` (an LLMResponse). Returns (weights over `allowed`, meta):
    meta = {attempts, retried, fallback, error, rationale, repairs, responses:[LLMResponse...]}.
    On total failure the weights are `prev_weights` (missing -> 0), i.e. hold."""
    meta: dict[str, Any] = {"attempts": 0, "retried": False, "fallback": False, "error": None,
                            "rationale": "", "repairs": {k: 0 for k in REPAIR_KEYS}, "responses": []}
    p = prompt
    for attempt in range(max_retries + 1):
        resp = ask(p)
        meta["attempts"] += 1
        meta["responses"].append(resp)
        try:
            res = parse_weights(getattr(resp, "text", str(resp)), allowed)
        except ParseError as e:
            meta["error"] = str(e)
            if attempt < max_retries:
                meta["retried"] = True
                p = prompt + RETRY_SUFFIX.format(error=str(e))
            continue
        meta["rationale"] = res.rationale
        for k, v in res.repairs.items():
            meta["repairs"][k] = meta["repairs"].get(k, 0) + v
        return res.weights, meta
    meta["fallback"] = True
    prev = prev_weights or {}
    return {a: float(prev.get(a, 0.0)) for a in allowed}, meta
