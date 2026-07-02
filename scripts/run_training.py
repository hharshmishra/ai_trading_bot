#!/usr/bin/env python
"""Manual trigger for the nightly meta-label + calibration training.

    python scripts/run_training.py            # train from logs/bitreinforcex.db
    python scripts/run_training.py --db path  # explicit database
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default=None, help="SQLite path (default: env/BITREINFORCEX_DB)")
    args = ap.parse_args()

    from persistence import Store, get_store
    from jobs.nightly import run_nightly_training

    store = Store(args.db) if args.db else get_store()
    summary = run_nightly_training(store)
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
