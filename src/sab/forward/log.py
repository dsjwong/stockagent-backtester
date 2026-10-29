"""Append-only decision log (JSON lines) with a hash chain.

Each record carries `prev_hash` (the previous record's `record_hash`) and its own `record_hash`
(sha256 over the canonical JSON of the record without `record_hash`). Rewriting or deleting any
earlier line breaks the chain, and git history of the file timestamps each append. The only write
path is `append`, which opens the file in 'a' mode, refuses a second (date, mode) record, refuses to
go back in time within a mode, and refuses dates before `min_date`.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

GENESIS = "0" * 64


class DuplicateDecision(RuntimeError):
    pass


class LogIntegrityError(RuntimeError):
    pass


def _hash(rec: dict[str, Any]) -> str:
    body = {k: v for k, v in rec.items() if k != "record_hash"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


class DecisionLog:
    def __init__(self, path: Path, min_date: str | None = None) -> None:
        self.path = Path(path)
        self.min_date = min_date

    def read(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        out = []
        for i, line in enumerate(self.path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise LogIntegrityError(f"{self.path}:{i} is not valid JSON: {e}") from e
        return out

    def verify(self) -> int:
        """Re-walk the hash chain; return the number of records or raise LogIntegrityError."""
        prev = GENESIS
        recs = self.read()
        for i, r in enumerate(recs, 1):
            if r.get("prev_hash") != prev:
                raise LogIntegrityError(f"{self.path}: record {i} prev_hash does not match the previous record")
            if r.get("record_hash") != _hash(r):
                raise LogIntegrityError(f"{self.path}: record {i} content does not match its record_hash")
            prev = r["record_hash"]
        return len(recs)

    def get(self, date: str, mode: str) -> dict[str, Any] | None:
        for r in self.read():
            if r["decision_date"] == date and r["mode"] == mode:
                return r
        return None

    def append(self, rec: dict[str, Any]) -> dict[str, Any]:
        for k in ("decision_date", "mode"):
            if k not in rec:
                raise ValueError(f"record needs {k!r}")
        existing = self.read()
        self.verify()
        if self.min_date is not None and rec["decision_date"] < self.min_date:
            raise ValueError(f"decision_date {rec['decision_date']} is before the forward start {self.min_date}")
        for r in existing:
            if r["decision_date"] == rec["decision_date"] and r["mode"] == rec["mode"]:
                raise DuplicateDecision(f"{rec['decision_date']}/{rec['mode']} already logged")
        same_mode = [r["decision_date"] for r in existing if r["mode"] == rec["mode"]]
        if same_mode and rec["decision_date"] < max(same_mode):
            raise ValueError(f"{rec['decision_date']} is earlier than the last {rec['mode']} record ({max(same_mode)})")
        full = {k: v for k, v in rec.items() if k not in ("prev_hash", "record_hash")}
        full["prev_hash"] = existing[-1]["record_hash"] if existing else GENESIS
        full["record_hash"] = _hash(full)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(full, sort_keys=True) + "\n")
            f.flush()
            os.fsync(f.fileno())
        return full
