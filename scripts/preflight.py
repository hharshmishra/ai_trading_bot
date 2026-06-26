#!/usr/bin/env python3
"""Preflight check (Phase 6): verify a deployment is coherent before starting.

    ./venv/bin/python scripts/preflight.py

Checks core imports, SQLite init + CRUD, the RAG embedder, and required env vars.
Exit 0 = ready, 1 = problems found.
"""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

ok = True


def check(name, fn):
    global ok
    try:
        print(f"  ok  {name}: {fn()}")
    except Exception as e:
        ok = False
        print(f"  XX  {name}: {type(e).__name__}: {e}")


print("BitReinforceX preflight\n")


def _imports():
    import ccxt, openai, telegram, langgraph, pandas, numpy, pandas_ta, feedparser, pydantic  # noqa
    import brain.decision_maker, cycle, signals, grader, persistence, rag, ingestion, telegram_app  # noqa
    return "core + app modules import"


def _db():
    from persistence import Store
    d = tempfile.mkdtemp()
    s = Store(os.path.join(d, "pf.db"))
    sid = s.create_session(pair="BTCUSDT", tf="4h")
    assert s.get_session(sid)
    s.close()
    return "SQLite schema + CRUD"


def _embedder():
    from rag import get_embedder
    e = get_embedder()
    e.embed(["preflight smoke test"])
    return f"{type(e).__name__} dim={e.dim}"


def _env():
    missing = [k for k in ("OPENAI_API_KEY", "TELEGRAM_BOT_TOKEN") if not os.getenv(k)]
    if missing:
        raise RuntimeError("missing required env: " + ", ".join(missing))
    chans = [k for k in ("CUSTOMER_CHAT_ID", "DEV_CHAT_ID",
                         "TELEGRAM_SIGNALS_CHANNEL_ID", "TELEGRAM_DEV_CHANNEL_ID") if os.getenv(k)]
    return f"required set; channels: {chans or 'NONE (no signals will send)'}"


check("imports", _imports)
check("database", _db)
check("rag embedder", _embedder)
check("env vars", _env)

print("\n" + ("READY" if ok else "NOT READY — fix the XX items above"))
sys.exit(0 if ok else 1)
