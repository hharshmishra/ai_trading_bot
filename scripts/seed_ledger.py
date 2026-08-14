#!/usr/bin/env python3
"""One-time evidence-ledger seed (v3.8 deploy step).

Builds logs/emission_ledger.json from the box's graded history so the
edge-first gate starts with earned cohorts instead of an empty table.
v3.8 rows carry candidate_trigger/candidate_action natively; legacy rows
(v3.7.1) are synthesized conservatively from the recorded funnel telemetry:

  gate_reason nwe_* / no_brain_agreement            -> ("nwe", nwe_action)
  low_volume with a directional nwe_action          -> ("nwe", nwe_action)
  conf_* / counter_trend_conf (directional final)   -> ("conf", final_action)
  meta_gate/conf_saturated: nwe_action if present,
    else conf when final_confidence cleared the gate -> best recoverable
  trend rows: only emitted ones carry a trustworthy direction (the final);
    suppressed trend candidates are skipped rather than guessed.

Usage:
  venv/bin/python scripts/seed_ledger.py            # dry run (prints table)
  venv/bin/python scripts/seed_ledger.py --yes      # write the artifact
  BITREINFORCEX_DB=... to point at a different DB.
"""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import config  # noqa: E402
from jobs.ledger import build_ledger, save_ledger  # noqa: E402
from persistence import Store  # noqa: E402


def synth_candidate(r: dict) -> dict:
    if r.get("candidate_trigger") and r.get("candidate_action"):
        return r
    reason = (r.get("gate_reason") or "").lower()
    nwe = (r.get("nwe_action") or "").lower()
    final = (r.get("final_action") or "").lower()
    conf = float(r.get("final_confidence") or 0.0)
    src = act = None
    if reason.startswith("nwe") or reason == "no_brain_agreement":
        src, act = "nwe", nwe
    elif reason == "low_volume" and nwe in ("buy", "sell"):
        src, act = "nwe", nwe
    elif reason.startswith("conf") or reason == "counter_trend_conf":
        src, act = "conf", final
    elif reason in ("meta_gate", "conf_saturated"):
        if nwe in ("buy", "sell"):
            src, act = "nwe", nwe
        elif final in ("buy", "sell") and conf >= config.CONFIDENCE_GATE:
            src, act = "conf", final
    elif (reason.startswith(("trend", "reversal", "counter_trend"))
          and r.get("emitted") and final in ("buy", "sell")):
        src, act = "trend", final
    if src and act in ("buy", "sell"):
        r = dict(r)
        r["candidate_trigger"], r["candidate_action"] = src, act
    return r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--yes", action="store_true", help="write the artifact")
    ap.add_argument("--dry-run", action="store_true", help="(default) print only")
    args = ap.parse_args()

    store = Store(os.getenv("BITREINFORCEX_DB", "logs/bitreinforcex.db"))
    rows = [synth_candidate(r) for r in store.training_rows()]
    led = build_ledger(rows)
    cohorts = led.get("cohorts") or {}

    print(f"graded rows: {len(rows)}  cohorts: {len(cohorts)}")
    floor, min_n = config.LEDGER_FLOOR, config.LEDGER_MIN_N
    print(f"posture: rate >= {floor:.2f} AND WilsonLB >= {config.LEDGER_LB_GUARD:.2f} at n >= {min_n}\n")
    print(f"{'cohort':45s} {'n':>5s} {'rate':>6s} {'LB':>6s}  emit?")
    for key in sorted(cohorts, key=lambda k: -cohorts[k]['lb']):
        c = cohorts[key]
        if c["n"] < 5:
            continue
        ok = "YES" if (c["n"] >= min_n and c["rate"] >= floor
                       and c["lb"] >= config.LEDGER_LB_GUARD) else "-"
        print(f"{key:45s} {c['n']:5d} {c['rate']:6.1%} {c['lb']:6.1%}  {ok}")

    if args.yes:
        save_ledger(led)
        print(f"\nwrote {config.LEDGER_PATH}")
    else:
        print("\ndry run — pass --yes to write")
    store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
