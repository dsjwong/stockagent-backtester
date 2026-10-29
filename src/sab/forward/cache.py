"""Prompt cache for forward days. The historical cache key is sha256(provider|model|T|seed|prompt);
that is already prompt-specific, but a forward answer must additionally be bound to the decision
date, the mode and the model, and verified on read, so a stale answer from another date/mode/model
can never be returned even if two prompts happened to coincide."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sab.forward.config import FORWARD_DIR
from sab.llm.cache import LLMCache


def forward_cache_key(date: str, mode: str, model: str, base_key: str) -> str:
    return hashlib.sha256(f"forward|{date}|{mode}|{model}|{base_key}".encode("utf-8")).hexdigest()


@dataclass
class ForwardCache(LLMCache):
    """LLMCache bound to one (decision date, mode, model). Filenames are the forward key and every
    record carries its binding, which `get` re-checks."""

    date: str = ""
    mode: str = ""
    model: str = ""

    def __post_init__(self) -> None:
        if not (self.date and self.mode and self.model):
            raise ValueError("ForwardCache needs date, mode and model")

    @property
    def binding(self) -> dict[str, str]:
        return {"date": self.date, "mode": self.mode, "model": self.model}

    def path(self, key: str) -> Path:
        return Path(self.directory) / f"{forward_cache_key(self.date, self.mode, self.model, key)}.json"

    def get(self, key: str) -> dict[str, Any] | None:
        n_hit = self.hits
        rec = super().get(key)
        if rec is not None and rec.get("forward") != self.binding:
            self._mem.pop(key, None)
            self.hits, self.misses = n_hit, self.misses + 1
            return None
        return rec

    def put(self, key: str, record: dict[str, Any]):
        return super().put(key, {**record, "forward": self.binding})


def default_cache_dir(directory: Path = FORWARD_DIR) -> Path:
    return Path(directory) / "cache"
