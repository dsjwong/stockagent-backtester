"""Content-addressed disk cache for LLM completions.

One JSON file per completion under data/llm_cache/<key>.json where
key = sha256("provider|model|temperature|seed|prompt"). The file stores the prompt, the
response text, usage, model, latency and a timestamp, so a cached backtest can be audited
prompt-by-prompt and replayed with zero provider calls (CLAUDE.md cost rule). The directory
is git-ignored; a run's config records the provider/model so its cache entries can be
regenerated deliberately.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

CACHE_DIR = Path(__file__).resolve().parents[3] / "data" / "llm_cache"


def cache_key(provider: str, model: str, prompt: str, temperature: float = 0.0, seed: int | None = None) -> str:
    raw = f"{provider}|{model}|{float(temperature):g}|{'' if seed is None else int(seed)}|{prompt}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass
class LLMCache:
    """Tiny file cache. `hits` / `misses` / `writes` counters make "zero provider calls on
    replay" testable."""

    directory: Path = CACHE_DIR
    hits: int = 0
    misses: int = 0
    writes: int = 0
    _mem: dict[str, dict[str, Any]] = field(default_factory=dict, repr=False)

    def path(self, key: str) -> Path:
        return Path(self.directory) / f"{key}.json"

    def get(self, key: str) -> dict[str, Any] | None:
        if key in self._mem:
            self.hits += 1
            return self._mem[key]
        p = self.path(key)
        if p.exists():
            try:
                rec = json.loads(p.read_text())
            except json.JSONDecodeError:            # half-written file: treat as a miss
                self.misses += 1
                return None
            self._mem[key] = rec
            self.hits += 1
            return rec
        self.misses += 1
        return None

    def put(self, key: str, record: dict[str, Any]) -> Path:
        p = self.path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        rec = {**record, "key": key, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(rec, indent=1))
        tmp.replace(p)
        self._mem[key] = rec
        self.writes += 1
        return p

    def __contains__(self, key: str) -> bool:
        return key in self._mem or self.path(key).exists()

    def __len__(self) -> int:
        d = Path(self.directory)
        return len(list(d.glob("*.json"))) if d.exists() else 0

    def reset_counters(self) -> None:
        self.hits = self.misses = self.writes = 0


DEFAULT_CACHE = LLMCache()
