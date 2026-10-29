"""Forward (live) paper trading: the clean leakage test.

Decisions are made on the newest closed bar, logged append-only, and committed before the next
open, so git timestamps prove they were made without knowledge of what followed.
"""

from sab.forward.cache import ForwardCache, forward_cache_key
from sab.forward.config import FORWARD_DIR, ForwardConfig
from sab.forward.data import ForwardDataError, fetch_incremental, latest_complete_bar_date, load_forward_market
from sab.forward.log import DecisionLog, DuplicateDecision, LogIntegrityError
from sab.forward.runner import ForwardAgent, ReplayStrategy, StaleDecision, decision_deadline, run_forward

__all__ = [
    "ForwardCache", "forward_cache_key", "FORWARD_DIR", "ForwardConfig", "ForwardDataError", "fetch_incremental",
    "latest_complete_bar_date", "load_forward_market", "DecisionLog", "DuplicateDecision", "LogIntegrityError",
    "ForwardAgent", "ReplayStrategy", "StaleDecision", "decision_deadline", "run_forward",
]
