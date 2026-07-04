#!/usr/bin/env python3
"""Reset the system's LEARNED state to zero — for a clean production go-live.

What this wipes (everything the system taught itself):
  * the four agent policies + the brain policy (agents re-seed their
    deterministic Random(42) defaults on next start)
  * the trading DB's learning tables — predictions / outcomes / rewards /
    sessions (the graded track record the RL trains on)
  * nightly-training artifacts — meta model, calibration knots, empirical-Bayes
    indicator confidences
  * per-agent jsonl line logs

What it KEEPS (inputs and business data, not learned):
  * data/ and data/history/ — cached market OHLCV (expensive to refetch)
  * macro_snapshots + the news corpus (news_items) — inputs, not learned
    (pass --wipe-news to clear the news corpus too)
  * ecosystems cache
  * logs/subscriptions.db — paying customers (NEVER touched)
  * .env

Everything wiped is first BACKED UP to logs/pre-reset-<UTC>/, so a reset is
reversible by copying the backup back.

Safe by default: prints a plan and exits (--dry-run is implied). Pass --yes to
execute. Refuses to run while telegram_app is live (unless --force) so it never
races the scheduler/grader writing to the same files.

    python scripts/reset_learning.py            # show the plan, change nothing
    python scripts/reset_learning.py --yes       # do it
    python scripts/reset_learning.py --yes --wipe-news   # also clear news corpus
"""
from __future__ import annotations

import argparse
import os
import shutil
import sqlite3
import subprocess
import sys

# scripts/ -> repo root on sys.path (cwd change happens only in __main__, so a
# test that chdir's into a tmp tree and reloads this module is not overridden)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import config  # noqa: E402  (loads .env; gives us the configured paths)

POLICY_FILES = [
    "logs/news_agent_policy.json",
    "logs/research_agent_policy.json",
    "logs/indicator_agent_policy.json",
    "logs/derivatives_agent_policy.json",
    "logs/brain_policy.json",
]
LINE_LOGS = [
    "logs/predictions_log.json",
    "logs/indicator_predictions.jsonl",
    "logs/research_predictions.jsonl",
]
POLICY_BACKUPS_GLOB = "logs"                    # *.bak-*dim live here
LEARNING_TABLES = ["predictions", "outcomes", "rewards", "sessions"]
NEWS_TABLES = ["news_items"]                    # only with --wipe-news


def _nightly_artifacts():
    # resolved at call time so a reload/monkeypatch of config is honoured
    return [config.META_MODEL_PATH, config.META_METRICS_PATH,
            config.CALIBRATION_PATH, config.INDICATOR_CONF_PATH]


def _db_path():
    return os.getenv("BITREINFORCEX_DB", "logs/bitreinforcex.db")


def _running() -> bool:
    try:
        out = subprocess.run(["pgrep", "-f", "telegram_app.py"],
                             capture_output=True, text=True)
        return out.returncode == 0 and out.stdout.strip() != ""
    except Exception:
        return False


def _files_present(paths):
    return [p for p in paths if p and os.path.exists(p)]


def _table_counts(db_path, tables):
    counts = {}
    if not os.path.exists(db_path):
        return counts
    con = sqlite3.connect(db_path)
    try:
        for t in tables:
            try:
                counts[t] = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            except sqlite3.OperationalError:
                pass                            # table absent — fine
    finally:
        con.close()
    return counts


def main() -> int:
    ap = argparse.ArgumentParser(description="Reset learned state to zero.")
    ap.add_argument("--yes", action="store_true", help="execute (default: dry run)")
    ap.add_argument("--wipe-news", action="store_true",
                    help="also clear the stored news corpus (news_items)")
    ap.add_argument("--force", action="store_true",
                    help="run even if telegram_app appears to be running")
    args = ap.parse_args()

    if _running() and not args.force:
        print("REFUSING: telegram_app.py looks like it is running. Stop it first "
              "(or pass --force). A reset must not race the live scheduler/grader.")
        return 2

    db_path = _db_path()
    tables = LEARNING_TABLES + (NEWS_TABLES if args.wipe_news else [])
    policy_files = _files_present(POLICY_FILES)
    backups = _files_present([os.path.join(POLICY_BACKUPS_GLOB, f)
                              for f in os.listdir(POLICY_BACKUPS_GLOB)
                              if ".bak-" in f]) if os.path.isdir(POLICY_BACKUPS_GLOB) else []
    artifacts = _files_present(_nightly_artifacts())
    line_logs = _files_present(LINE_LOGS)
    counts = _table_counts(db_path, tables)

    print("=" * 64)
    print("RESET LEARNING TO ZERO" + ("" if args.yes else "   (DRY RUN — nothing will change)"))
    print("=" * 64)
    print("\nWILL DELETE (after backup):")
    for p in policy_files + backups + artifacts + line_logs:
        print(f"   file   {p}")
    for t, n in counts.items():
        print(f"   table  {db_path}:{t}  ({n} rows -> 0)")
    print("\nWILL KEEP (untouched):")
    for keep in ("data/  (market OHLCV cache)", f"{config.MEMBERSHIP_DB}  (customers)",
                 "macro_snapshots" + ("" if args.wipe_news else " + news corpus"),
                 config.ECOSYSTEMS_CACHE_PATH, ".env"):
        print(f"   {keep}")

    if not args.yes:
        print("\nDry run. Re-run with --yes to execute.")
        return 0

    # ---- backup ----
    ts = subprocess.run(["date", "-u", "+%Y%m%dT%H%M%SZ"],
                        capture_output=True, text=True).stdout.strip() or "reset"
    backup_dir = os.path.join("logs", f"pre-reset-{ts}")
    os.makedirs(backup_dir, exist_ok=True)
    for p in policy_files + backups + artifacts + line_logs:
        try:
            shutil.copy2(p, os.path.join(backup_dir, os.path.basename(p)))
        except Exception as e:
            print(f"   ! backup failed for {p}: {e}")
    if os.path.exists(db_path):
        shutil.copy2(db_path, os.path.join(backup_dir, os.path.basename(db_path)))
    print(f"\nBacked up to {backup_dir}/")

    # ---- wipe files ----
    for p in policy_files + backups + artifacts + line_logs:
        try:
            os.remove(p)
        except Exception as e:
            print(f"   ! could not delete {p}: {e}")

    # ---- wipe DB tables (keep schema; VACUUM) ----
    if os.path.exists(db_path):
        con = sqlite3.connect(db_path)
        try:
            for t in tables:
                try:
                    con.execute(f"DELETE FROM {t}")
                except sqlite3.OperationalError:
                    pass
            con.commit()
            con.execute("VACUUM")
        finally:
            con.close()

    print("\nDONE. The next start re-seeds default policies and begins learning "
          "from zero. Restore: copy the backup files back and re-import the DB.")
    return 0


if __name__ == "__main__":
    os.chdir(ROOT)          # resolve logs/ + data/ against the repo root
    raise SystemExit(main())
