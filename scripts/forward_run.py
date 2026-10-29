"""Forward paper trading, one run = one trading day: incremental fetch -> decide (anon AND real) -> log.

    .venv/bin/python scripts/forward_run.py                 # live: fetch, decide on the newest closed bar, append to forward/decisions.jsonl
    .venv/bin/python scripts/forward_run.py --dry-run       # proof of life on the newest closed bar; writes forward/dryrun/ (git-ignored), never forward/decisions.jsonl
    .venv/bin/python scripts/forward_run.py --no-fetch      # decide from already-stored bars

The pinned model/prompt/cadence live in forward/config.json (created on first live run with --start,
default 2026-10-30). The first decision is the first bar >= start (entered from cash); then the first bar of
each ISO week, like the historical weekly agent. A decision is refused after the next US open (it would no
longer be "in advance"). Re-running a day replays the log: no model call, no duplicate record.

Exit codes: 0 ok / nothing to do, 2 model quota exhausted (re-run before the next open), 3 data problem or stale.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sab.forward import (FORWARD_DIR, ForwardConfig, ForwardDataError, StaleDecision, fetch_incremental,  # noqa: E402
                         latest_complete_bar_date, load_forward_market, run_forward)
from sab.llm.providers import QuotaExhausted  # noqa: E402

DEFAULT_START = "2026-10-30"      # next trading day after 2026-10-29, when the forward test was set up


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default=None, help=f"forward start date (only used to create forward/config.json; default {DEFAULT_START})")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-fetch", action="store_true")
    ap.add_argument("--modes", nargs="+", default=["anon", "real"], choices=["anon", "real"])
    args = ap.parse_args(argv)

    now = dt.datetime.now(dt.timezone.utc)
    have_cfg = (FORWARD_DIR / "config.json").exists()
    if args.dry_run:
        out_dir = FORWARD_DIR / "dryrun"
        out_dir.mkdir(parents=True, exist_ok=True)
        start = latest_complete_bar_date(now).isoformat()
        base = ForwardConfig.load_or_create(FORWARD_DIR) if have_cfg else ForwardConfig(start=start)
        cfg = ForwardConfig(**{**asdict(base), "start": start})
        (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2) + "\n")
    else:
        out_dir = FORWARD_DIR
        cfg = ForwardConfig.load_or_create(FORWARD_DIR, start=args.start or (None if have_cfg else DEFAULT_START))

    try:
        if not args.no_fetch:
            added = fetch_incremental(now, FORWARD_DIR)
            print(f"fetch: +{sum(added.values())} bars ({', '.join(f'{k}:{v}' for k, v in added.items() if v) or 'nothing new'})")
        market = load_forward_market(FORWARD_DIR)
        last = market.dates[-1].date()
        print(f"market: {len(market)} bars through {last}; forward start {cfg.start}; model {cfg.model}; dry_run={args.dry_run}")
        if str(last) < cfg.start:
            print("no forward bars yet (newest bar is before the start date): nothing to decide")
            return 0
        new = run_forward(cfg, market, directory=out_dir, modes=args.modes, dry_run=args.dry_run,
                          enforce_deadline=not args.dry_run, now=lambda: dt.datetime.now(dt.timezone.utc))
    except QuotaExhausted as e:
        print(f"!! QUOTA EXHAUSTED: {e}\n   re-run before the next US open; completed modes are already logged.")
        return 2
    except (ForwardDataError, StaleDecision) as e:
        print(f"!! {type(e).__name__}: {e}")
        return 3
    if not new:
        print(f"nothing to decide for {last} (not a decision bar, or already logged)")
    for r in new:
        top = sorted(r["weights"].items(), key=lambda kv: -kv[1])[:5]
        print(f"DECIDED {r['decision_date']} {r['mode']:4s} model_served={r['model_served']} exposure={r['exposure']:.2f} "
              f"prompt={r['prompt_sha256'][:12]} top={top} fallback={r['fallback']}\n    rationale: {r['rationale']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
