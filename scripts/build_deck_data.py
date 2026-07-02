#!/usr/bin/env python
"""Assemble the evidence JSON for docs/accuracy-upgrade.html and inject it.

Reads archived backtest reports (baseline + gate-v2 candidate) and, when a
live DB exists, shadow stats (meta_p / calibrated_conf coverage, regime mix).
Injects the JSON into the deck between the markers:

    /*__DECK_DATA_START__*/ ... /*__DECK_DATA_END__*/

so the deck stays a single self-contained file whose numbers are traceable to
the archived report JSONs.

    python scripts/build_deck_data.py \
        --baseline logs/backtest/baseline/report.json \
        --candidate logs/backtest/gate-v2/report.json \
        --deck docs/accuracy-upgrade.html
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

START = "/*__DECK_DATA_START__*/"
END = "/*__DECK_DATA_END__*/"


def _load(path):
    if not path or not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _db_shadow_stats(db_path: str):
    if not os.path.exists(db_path):
        return None
    import sqlite3
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            "SELECT COUNT(*) AS n, SUM(graded) AS graded, "
            "SUM(CASE WHEN meta_p IS NOT NULL THEN 1 ELSE 0 END) AS with_meta, "
            "SUM(CASE WHEN regime IS NOT NULL THEN 1 ELSE 0 END) AS with_regime "
            "FROM predictions").fetchone()
        regimes = {r["regime"] or "none": r["c"] for r in conn.execute(
            "SELECT regime, COUNT(*) AS c FROM predictions GROUP BY regime")}
        return {"predictions": row["n"], "graded": row["graded"] or 0,
                "with_meta": row["with_meta"] or 0, "with_regime": row["with_regime"] or 0,
                "regime_mix": regimes}
    except Exception:
        return None
    finally:
        conn.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline", default="logs/backtest/baseline/report.json")
    ap.add_argument("--candidate", default=None)
    ap.add_argument("--smoke", default="logs/backtest/smoke/report.json")
    ap.add_argument("--db", default=os.getenv("BITREINFORCEX_DB", "logs/bitreinforcex.db"))
    ap.add_argument("--deck", default="docs/accuracy-upgrade.html")
    ap.add_argument("--out-json", default=None, help="also write the raw JSON here")
    args = ap.parse_args()

    data = {
        "baseline": _load(args.baseline),
        "candidate": _load(args.candidate),
        "smoke": _load(args.smoke),
        "meta_metrics": _load("logs/meta_metrics.json"),
        "shadow": _db_shadow_stats(args.db),
    }
    payload = json.dumps(data, default=str)

    if args.out_json:
        with open(args.out_json, "w") as f:
            f.write(payload)

    if os.path.exists(args.deck):
        with open(args.deck) as f:
            html = f.read()
        pattern = re.escape(START) + r".*?" + re.escape(END)
        replacement = f"{START}{payload}{END}"
        new_html, n = re.subn(pattern, lambda _: replacement, html, flags=re.S)
        if n == 0:
            print(f"[deck-data] markers not found in {args.deck}; nothing injected")
            return 1
        with open(args.deck, "w") as f:
            f.write(new_html)
        print(f"[deck-data] injected {len(payload)} bytes into {args.deck}")
    else:
        print(f"[deck-data] {args.deck} missing; JSON only")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
