"""Model providers behind one function: `complete(provider, model, prompt, ...) -> LLMResponse`.

Offline (deterministic, used by every test):
* `mock`        -- weights derived from sha256(prompt): same prompt -> same answer, different
                   prompt -> different answer. Ignores temperature/seed.
* `mock_ew`     -- always equal weights over the tickers named in the prompt (sum = 1).
* `mock_random` -- weights from a RNG seeded by (seed, prompt); with seed=None every call draws
                   a fresh seed (mimics a temperature>0 model), so it is the "inconsistent" model
                   for consistency checks.
With provider="mock", model="mock_ew" / "mock_random" selects the variant (so `--provider mock --model mock_ew`
works). The mocks read the ticker list from the prompt's `Tickers:` line (see prompt.py), so they answer
in whatever namespace (real or anonymised) the prompt uses.

Hosted / local (thin adapters, NOT exercised in M4a -- no network in tests):
* `anthropic` (ANTHROPIC_API_KEY), `openai` (OPENAI_API_KEY), `gemini` (GEMINI_API_KEY or
  GOOGLE_API_KEY; REST via `requests`, paced + 429 backoff, see `_gemini`), `ollama`
  (OLLAMA_HOST, default http://localhost:11434, uses `requests`).
  SDKs are imported lazily inside the adapter, credentials are read from the environment after
  an optional `.env` load (python-dotenv), and a missing SDK raises ImportError with the pip
  extra to install (`pip install -e '.[llm]'`). `gemini` and `ollama` need no SDK.

`complete()` consults the disk cache first (sab.llm.cache) unless `bypass_cache=True`; every
fresh response is written back, so a replay never calls a provider.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from sab.llm.cache import DEFAULT_CACHE, LLMCache, cache_key

MOCK_PROVIDERS = ("mock", "mock_ew", "mock_random")
REAL_PROVIDERS = ("anthropic", "openai", "gemini", "ollama", "openai_compat")
PROVIDERS = MOCK_PROVIDERS + REAL_PROVIDERS


@dataclass
class LLMResponse:
    text: str
    usage: dict[str, int] = field(default_factory=dict)     # prompt_tokens, completion_tokens
    model: str = ""
    latency_s: float = 0.0
    provider: str = ""
    cached: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- helpers

_TICKER_LINE = re.compile(r"^Tickers:\s*(.+)$", re.MULTILINE)


def tickers_from_prompt(prompt: str) -> list[str]:
    """The mocks' only view of the universe: the `Tickers: A, B, C` line the prompt builder writes."""
    m = _TICKER_LINE.search(prompt)
    if not m:
        raise ValueError("mock provider: prompt has no 'Tickers:' line")
    return [t.strip() for t in m.group(1).split(",") if t.strip()]


def approx_tokens(text: str) -> int:
    """~4 characters per token; the same estimate `--estimate-tokens` uses."""
    return max(1, len(text) // 4)


def _weights_json(tickers: list[str], weights: list[float], rationale: str) -> str:
    return json.dumps({"weights": {t: float(w) for t, w in zip(tickers, weights)}, "rationale": rationale})


def _mock_response(text: str, prompt: str, model: str, provider: str) -> LLMResponse:
    return LLMResponse(text=text, usage={"prompt_tokens": approx_tokens(prompt), "completion_tokens": approx_tokens(text)},
                       model=model, latency_s=0.0, provider=provider)


# --------------------------------------------------------------------------- mock providers

def _mock(model: str, prompt: str, temperature: float, max_tokens: int, seed: int | None) -> LLMResponse:
    tickers = tickers_from_prompt(prompt)
    digest = hashlib.sha256(prompt.encode("utf-8")).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
    k = int(rng.integers(1, len(tickers) + 1))                       # how many names to hold
    picks = rng.choice(len(tickers), size=k, replace=False)
    raw = rng.dirichlet(np.ones(k)) * float(rng.uniform(0.6, 1.0))    # sum in [0.6, 1.0]
    w = np.zeros(len(tickers))
    w[picks] = np.floor(raw * 1e4) / 1e4                              # 4 dp, floors keep sum <= 1
    return _mock_response(_weights_json(tickers, w.tolist(), f"mock: hash-derived weights over {k} names"),
                          prompt, model, "mock")


def _mock_ew(model: str, prompt: str, temperature: float, max_tokens: int, seed: int | None) -> LLMResponse:
    tickers = tickers_from_prompt(prompt)
    w = [1.0 / len(tickers)] * len(tickers)
    return _mock_response(_weights_json(tickers, w, "mock_ew: equal weight every name"), prompt, model, "mock_ew")


_random_counter = itertools.count()


def _mock_random(model: str, prompt: str, temperature: float, max_tokens: int, seed: int | None) -> LLMResponse:
    tickers = tickers_from_prompt(prompt)
    draw = int(seed) if seed is not None else next(_random_counter)
    digest = hashlib.sha256(prompt.encode("utf-8")).digest()
    rng = np.random.default_rng([draw, int.from_bytes(digest[:8], "big")])
    raw = rng.dirichlet(np.ones(len(tickers))) * float(rng.uniform(0.5, 1.0))
    w = np.floor(raw * 1e4) / 1e4
    return _mock_response(_weights_json(tickers, w.tolist(), f"mock_random: seeded draw {draw}"), prompt, model, "mock_random")


# --------------------------------------------------------------------------- real adapters (lazy)

_ENV_FILE = Path(__file__).resolve().parents[3] / ".env"


def _load_env() -> None:
    try:
        from dotenv import load_dotenv     # optional; a plain environment works too
        load_dotenv(_ENV_FILE)             # explicit path: find_dotenv() breaks under stdin/-c
    except ImportError:
        pass


def _need(pkg: str, extra: str = "llm"):
    raise ImportError(f"provider needs the '{pkg}' package: pip install -e '.[{extra}]' (or pip install {pkg})")


def _env_key(*names: str) -> str:
    _load_env()
    for n in names:
        v = os.environ.get(n)
        if v:
            return v
    raise RuntimeError(f"missing credential: set one of {names} in the environment or .env")


_NO_SAMPLING_PARAMS = re.compile(r"opus-(4-[78]|5)|sonnet-5|fable|mythos")   # these models reject `temperature` (400)


def _anthropic(model: str, prompt: str, temperature: float, max_tokens: int, seed: int | None) -> LLMResponse:
    try:
        import anthropic
    except ImportError:
        _need("anthropic")
    client = anthropic.Anthropic(api_key=_env_key("ANTHROPIC_API_KEY"))
    kw: dict[str, Any] = {} if _NO_SAMPLING_PARAMS.search(model) else {"temperature": temperature}
    t0 = time.perf_counter()
    msg = client.messages.create(model=model, max_tokens=max_tokens, messages=[{"role": "user", "content": prompt}], **kw)
    text = "".join(getattr(b, "text", "") for b in msg.content)
    return LLMResponse(text=text, usage={"prompt_tokens": msg.usage.input_tokens, "completion_tokens": msg.usage.output_tokens},
                       model=msg.model, latency_s=time.perf_counter() - t0, provider="anthropic")


def _openai(model: str, prompt: str, temperature: float, max_tokens: int, seed: int | None) -> LLMResponse:
    try:
        import openai
    except ImportError:
        _need("openai")
    client = openai.OpenAI(api_key=_env_key("OPENAI_API_KEY"))
    kw: dict[str, Any] = {"model": model, "messages": [{"role": "user", "content": prompt}],
                          "temperature": temperature, "max_tokens": max_tokens}
    if seed is not None:
        kw["seed"] = int(seed)
    t0 = time.perf_counter()
    r = client.chat.completions.create(**kw)
    u = r.usage
    return LLMResponse(text=r.choices[0].message.content or "",
                       usage={"prompt_tokens": getattr(u, "prompt_tokens", 0), "completion_tokens": getattr(u, "completion_tokens", 0)},
                       model=r.model, latency_s=time.perf_counter() - t0, provider="openai")


class QuotaExhausted(RuntimeError):
    """A provider said the quota is gone for the day (or retries ran out on 429s). The run
    script catches this, reports which decisions completed (they are all in the cache) and
    stops -- it never burns the remaining quota on blind retries."""


GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
_last_call_at: dict[str, float] = {}


def _pace(provider: str, min_interval: float) -> None:
    """Sleep so that consecutive calls to `provider` are at least `min_interval` seconds apart
    (free tiers are rate-limited per minute; 6.5 s ~ 9 requests/min)."""
    last = _last_call_at.get(provider)
    if last is not None and min_interval > 0:
        wait = min_interval - (time.monotonic() - last)
        if wait > 0:
            time.sleep(wait)
    _last_call_at[provider] = time.monotonic()


def _gemini_retry_delay(r) -> float | None:
    """Seconds to wait as advertised by a 429: `Retry-After` header, else the RetryInfo detail."""
    ra = r.headers.get("retry-after")
    if ra:
        try:
            return float(ra)
        except ValueError:
            pass
    try:
        for d in r.json().get("error", {}).get("details", []):
            rd = d.get("retryDelay")
            if rd:
                return float(str(rd).rstrip("s"))
    except Exception:
        pass
    return None


def _gemini_is_daily(r) -> bool:
    try:
        err = r.json().get("error", {})
    except Exception:
        return False
    blob = json.dumps(err).lower()
    return "perday" in blob or "per day" in blob or "daily" in blob


def _gemini(model: str, prompt: str, temperature: float, max_tokens: int, seed: int | None) -> LLMResponse:
    """REST adapter (no SDK; `requests` is a core dependency). Free-tier discipline:
    * paced: >= SAB_GEMINI_MIN_INTERVAL seconds between calls (default 4.5 -> ~13/min; free tier is 15 RPM on flash-lite, 5 RPM on flash),
    * 429 -> honours Retry-After / RetryInfo, else exponential backoff (2, 4, 8, ... s),
      at most SAB_GEMINI_MAX_RETRIES (default 6) times; a 429 that names a daily quota, or
      exhausted retries, raises QuotaExhausted so the caller stops instead of burning quota,
    * 5xx and network errors (proxy/reset/timeout) -> same backoff; other 4xx -> RuntimeError with the API message.
    Thinking is set to SAB_GEMINI_THINKING (default `minimal`, Gemini 3 `thinkingLevel`; set to
    `off` to send no thinkingConfig) so `max_tokens` is spent on the answer, not on thoughts.
    Returned text = concatenation of non-thought parts."""
    import requests   # core dependency
    key = _env_key("GEMINI_API_KEY", "GOOGLE_API_KEY")
    min_interval = float(os.environ.get("SAB_GEMINI_MIN_INTERVAL", "4.5"))
    max_retries = int(os.environ.get("SAB_GEMINI_MAX_RETRIES", "6"))
    thinking = os.environ.get("SAB_GEMINI_THINKING", "minimal")
    gen: dict[str, Any] = {"temperature": temperature, "maxOutputTokens": max_tokens}
    if seed is not None:
        gen["seed"] = int(seed)
    if thinking and thinking.lower() != "off":
        gen["thinkingConfig"] = {"thinkingLevel": thinking}
    body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}], "generationConfig": gen}
    url = GEMINI_URL.format(model=model)
    headers = {"x-goog-api-key": key, "Content-Type": "application/json"}
    t0 = time.perf_counter()
    for attempt in range(max_retries + 1):
        _pace("gemini", min_interval)
        try:
            r = requests.post(url, headers=headers, json=body, timeout=float(os.environ.get("SAB_GEMINI_TIMEOUT", "120")))
        except requests.exceptions.RequestException as e:      # proxy hiccup, reset, timeout: back off and retry
            if attempt == max_retries:
                raise RuntimeError(f"gemini {model}: network error after {max_retries} retries: {e}") from e
            time.sleep(float(2 ** (attempt + 1)))
            continue
        if r.status_code == 200:
            break
        if r.status_code == 429:
            if _gemini_is_daily(r):
                raise QuotaExhausted(f"gemini {model}: daily quota exhausted (429): {_api_message(r)}")
            if attempt == max_retries:
                raise QuotaExhausted(f"gemini {model}: still 429 after {max_retries} retries: {_api_message(r)}")
            delay = _gemini_retry_delay(r) or float(2 ** (attempt + 1))
            time.sleep(min(delay, 120.0) + 1.0)
            continue
        if 500 <= r.status_code < 600 and attempt < max_retries:
            time.sleep(float(2 ** (attempt + 1)))
            continue
        raise RuntimeError(f"gemini {model}: HTTP {r.status_code}: {_api_message(r)}")
    d = r.json()
    cands = d.get("candidates") or []
    parts = (cands[0].get("content") or {}).get("parts", []) if cands else []
    text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
    um = d.get("usageMetadata", {}) or {}
    return LLMResponse(text=text,
                       usage={"prompt_tokens": int(um.get("promptTokenCount", 0) or 0),
                              "completion_tokens": int(um.get("candidatesTokenCount", 0) or 0),
                              "thought_tokens": int(um.get("thoughtsTokenCount", 0) or 0),
                              "finish_reason": (cands[0].get("finishReason", "") if cands else "NO_CANDIDATE")},
                       model=d.get("modelVersion", model), latency_s=time.perf_counter() - t0, provider="gemini")


def _api_message(r) -> str:
    """Message plus the quota ids / retry delay from the error details (what the 429 is really about)."""
    try:
        err = r.json().get("error", {})
        extra = []
        for d in err.get("details", []):
            for v in d.get("violations", []) or []:
                extra.append(f"quotaId={v.get('quotaId')} metric={v.get('quotaMetric')}")
            if d.get("retryDelay"):
                extra.append(f"retryDelay={d['retryDelay']}")
        return str(err.get("message", r.text[:300])) + (" [" + "; ".join(extra) + "]" if extra else "")
    except Exception:
        return r.text[:300]


def _ollama(model: str, prompt: str, temperature: float, max_tokens: int, seed: int | None) -> LLMResponse:
    import requests   # core dependency
    _load_env()
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    options: dict[str, Any] = {"temperature": temperature, "num_predict": max_tokens}
    if seed is not None:
        options["seed"] = int(seed)
    t0 = time.perf_counter()
    r = requests.post(f"{host}/api/generate", json={"model": model, "prompt": prompt, "stream": False, "options": options},
                      timeout=float(os.environ.get("OLLAMA_TIMEOUT", "300")))
    r.raise_for_status()
    body = r.json()
    return LLMResponse(text=body.get("response", ""),
                       usage={"prompt_tokens": int(body.get("prompt_eval_count", 0) or 0),
                              "completion_tokens": int(body.get("eval_count", 0) or 0)},
                       model=body.get("model", model), latency_s=time.perf_counter() - t0, provider="ollama")


def _openai_compat(model: str, prompt: str, temperature: float, max_tokens: int, seed: int | None) -> LLMResponse:
    """Any OpenAI-compatible chat endpoint (ollama `/v1`, LM Studio, `mlx_lm.server`, llama.cpp server, vLLM).
    Base URL from SAB_OPENAI_COMPAT_BASE_URL (e.g. http://localhost:11434/v1, http://localhost:1234/v1,
    http://localhost:8080/v1); optional SAB_OPENAI_COMPAT_API_KEY (most local servers ignore it);
    SAB_OPENAI_COMPAT_TIMEOUT seconds (default 300). Plain `requests`, no SDK. No default URL on purpose:
    the cross-family check must be pointed at a model deliberately."""
    import requests   # core dependency
    _load_env()
    base = os.environ.get("SAB_OPENAI_COMPAT_BASE_URL", "").rstrip("/")
    if not base:
        raise RuntimeError("openai_compat: set SAB_OPENAI_COMPAT_BASE_URL (e.g. http://localhost:11434/v1)")
    headers = {"Content-Type": "application/json"}
    if os.environ.get("SAB_OPENAI_COMPAT_API_KEY"):
        headers["Authorization"] = "Bearer " + os.environ["SAB_OPENAI_COMPAT_API_KEY"]
    body: dict[str, Any] = {"model": model, "messages": [{"role": "user", "content": prompt}],
                            "temperature": temperature, "max_tokens": max_tokens, "stream": False}
    if seed is not None:
        body["seed"] = int(seed)
    t0 = time.perf_counter()
    r = requests.post(f"{base}/chat/completions", headers=headers, json=body,
                      timeout=float(os.environ.get("SAB_OPENAI_COMPAT_TIMEOUT", "300")))
    if r.status_code != 200:
        raise RuntimeError(f"openai_compat {model} @ {base}: HTTP {r.status_code}: {r.text[:300]}")
    d = r.json()
    choices = d.get("choices") or []
    text = ((choices[0].get("message") or {}).get("content") or "") if choices else ""
    u = d.get("usage") or {}
    return LLMResponse(text=text, usage={"prompt_tokens": int(u.get("prompt_tokens", 0) or 0),
                                         "completion_tokens": int(u.get("completion_tokens", 0) or 0),
                                         "finish_reason": (choices[0].get("finish_reason") or "") if choices else "NO_CHOICE"},
                       model=d.get("model", model), latency_s=time.perf_counter() - t0, provider="openai_compat")


_ADAPTERS: dict[str, Callable[[str, str, float, int, int | None], LLMResponse]] = {
    "mock": _mock, "mock_ew": _mock_ew, "mock_random": _mock_random,
    "anthropic": _anthropic, "openai": _openai, "gemini": _gemini, "ollama": _ollama, "openai_compat": _openai_compat,
}


# --------------------------------------------------------------------------- entry point

def complete(provider: str, model: str, prompt: str, *, temperature: float = 0.0, max_tokens: int = 400,
             seed: int | None = None, cache: LLMCache | None = DEFAULT_CACHE, bypass_cache: bool = False) -> LLMResponse:
    """Return the model's completion for `prompt`, from the disk cache when the identical
    (provider, model, temperature, seed, prompt) was answered before. `cache=None` or
    `bypass_cache=True` skips the read; fresh responses are still written when a cache is given."""
    if provider not in _ADAPTERS:
        raise ValueError(f"unknown provider {provider!r}; choose from {PROVIDERS}")
    key = cache_key(provider, model, prompt, temperature, seed)
    if cache is not None and not bypass_cache:
        rec = cache.get(key)
        if rec is not None:
            r = rec["response"]
            return LLMResponse(text=r["text"], usage=dict(r.get("usage", {})), model=r.get("model", model),
                               latency_s=float(r.get("latency_s", 0.0)), provider=provider, cached=True)
    # `--provider mock --model mock_ew` and `--provider mock_ew` mean the same thing
    impl = model if provider == "mock" and model in MOCK_PROVIDERS else provider
    resp = _ADAPTERS[impl](model, prompt, float(temperature), int(max_tokens), seed)
    resp.provider = provider
    if cache is not None:
        cache.put(key, {"provider": provider, "model": model, "temperature": float(temperature), "seed": seed,
                        "prompt": prompt, "response": resp.to_dict(), "usage": resp.usage})
    return resp
