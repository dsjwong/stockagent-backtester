"""Pinned configuration of the forward test. Frozen into forward/config.json on first use and
re-checked on every run, so the model / prompt format / cadence cannot silently drift."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

FORWARD_DIR = Path(__file__).resolve().parents[3] / "forward"

PINNED_PROVIDER = "gemini"
PINNED_MODEL = "gemini-3.5-flash-lite"      # the historical study's model id (reports/runs/llm/*/config.json)


@dataclass(frozen=True)
class ForwardConfig:
    start: str                               # first forward decision is the first bar >= start
    provider: str = PINNED_PROVIDER
    model: str = PINNED_MODEL
    rebalance: str = "weekly"                # same cadence as the historical agent
    lookback: int = 60
    temperature: float = 0.0
    seed: int = 0
    max_tokens: int = 400
    commission_bps: float = 10.0
    slippage_bps: float = 5.0
    initial_cash: float = 1_000_000.0

    @classmethod
    def load_or_create(cls, directory: Path, start: str | None = None) -> "ForwardConfig":
        """Load directory/config.json; create it (needs `start`) if absent. An existing file wins:
        a different `start` is an error, never an overwrite."""
        p = Path(directory) / "config.json"
        if p.exists():
            cfg = cls(**json.loads(p.read_text()))
            if start is not None and start != cfg.start:
                raise ValueError(f"{p} pins start={cfg.start}; refusing to run with start={start}")
            return cfg
        if start is None:
            raise ValueError(f"{p} missing and no start date given")
        cfg = cls(start=start)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(asdict(cfg), indent=2) + "\n")
        return cfg
