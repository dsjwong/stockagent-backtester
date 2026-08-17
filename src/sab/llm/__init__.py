"""LLM trading agents (M4).

Pipeline per decision bar:  Context -> build_prompt -> complete (disk-cached) -> parse_weights
-> target_orders.  Everything here is offline-testable with the deterministic `mock*`
providers; the real providers (anthropic / openai / gemini / ollama) are thin adapters that
lazily import their SDK, so importing this package never needs any of them installed.

Cost rule (CLAUDE.md): every (provider, model, prompt, temperature, seed) -> response pair is
cached under data/llm_cache/, so replaying a backtest makes zero provider calls.
"""

from sab.llm.cache import CACHE_DIR, LLMCache, cache_key
from sab.llm.parse import ParseResult, parse_weights
from sab.llm.prompt import AnonMapping, PromptSpec, assert_prompt_no_future, build_prompt
from sab.llm.providers import PROVIDERS, LLMResponse, complete
from sab.llm.strategy import LLMAgent, consistency_check

__all__ = [
    "CACHE_DIR", "LLMCache", "cache_key", "ParseResult", "parse_weights", "AnonMapping", "PromptSpec",
    "assert_prompt_no_future", "build_prompt", "PROVIDERS", "LLMResponse", "complete", "LLMAgent",
    "consistency_check",
]
