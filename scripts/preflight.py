#!/usr/bin/env python3
"""Preflight check (Phase 6): verify a deployment is coherent before starting.

    ./venv/bin/python scripts/preflight.py

Checks core imports, SQLite init + CRUD, the RAG embedder, required env vars,
and .env/.env.example key parity.
Exit 0 = ready, 1 = problems found.
"""
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
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
    import config, backtest.engine, grading.barriers, jobs.nightly, agents.regime_agent, agents.derivatives_agent, agents.sentiment_agent  # noqa
    import backtest.data, backtest.metrics, backtest.report, backtest.sweep  # noqa
    import membership.store, membership.bot, membership.gate, membership.jobs, membership.payments, membership.plans  # noqa
    return "core + app + accuracy-v2 + membership modules import"


def _sklearn():
    import numpy, sklearn, joblib  # noqa
    if not numpy.__version__.startswith("1."):
        raise RuntimeError(f"numpy {numpy.__version__} breaks the pandas 2.0.3 / "
                           f"vendored pandas_ta ABI — pin numpy==1.25.0")
    from sklearn.linear_model import LogisticRegression  # noqa
    from sklearn.isotonic import IsotonicRegression  # noqa
    return f"sklearn {sklearn.__version__} on numpy {numpy.__version__} (train-side)"


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


def _env_parity(example=None):
    """Every key in .env.example must be present in the live environment.

    An ABSENT flag silently falls back to its code default — that is how the
    v3.5 sentiment voter sat dead in production for 19 days: SENTIMENT_ENABLED
    reached .env.example but never the box's .env, so the 5th voter cast zero
    votes with nothing anywhere reporting it. PRESENCE is the test, not
    truthiness (``T2_EXTRA_VOTES=`` is a legitimate empty value). WARN rather
    than fail — a deployment may omit optional keys deliberately. Key names
    only; values are never read or printed.
    """
    path = Path(example) if example else ROOT / ".env.example"
    if not path.exists():
        return f"skipped ({path.name} not found)"
    keys = []
    for line in path.read_text().splitlines():
        if line.lstrip().startswith("#") or "=" not in line:
            continue
        k = line.split("=", 1)[0].strip()
        if k and k.isupper() and " " not in k and k not in keys:
            keys.append(k)
    missing = [k for k in keys if k not in os.environ]
    if missing:
        return (f"WARN: {len(missing)}/{len(keys)} key(s) from {path.name} absent "
                f"from the environment — running on code defaults: "
                + ", ".join(missing))
    return f"{len(keys)} keys from {path.name} all present"


def _membership():
    """Rent-out mode misconfig fails BEFORE start, not at the first /grant.
    Token missing is a WARN (storefront degrades by design, Pro gate stays on);
    empty ADMIN_USER_IDS while enabled is a hard fail (no one can operate)."""
    import config as _c
    if not _c.MEMBERSHIP_ENABLED:
        return "disabled (flag off)"
    if not _c.ADMIN_USER_IDS:
        raise RuntimeError("MEMBERSHIP_ENABLED=true but ADMIN_USER_IDS is empty")
    notes = []
    if not os.getenv("MEMBERSHIP_BOT_TOKEN"):
        notes.append("WARN: no MEMBERSHIP_BOT_TOKEN (storefront bot won't start; gate still on)")
    return f"enabled, {len(_c.ADMIN_USER_IDS)} admin(s)" + ("; " + "; ".join(notes) if notes else "")


def _ledger():
    """v3.8 edge-first gate: without the ledger artifact NOTHING emits except
    new-source probation — a fresh deploy that skipped scripts/seed_ledger.py
    would go silent for up to 24h (until the nightly rebuild). WARN, not fail:
    the artifact self-heals nightly."""
    import config as _c
    if not _c.EMISSION_V2_ENABLED:
        return "disabled (EMISSION_V2_ENABLED off)"
    if not os.path.exists(_c.LEDGER_PATH):
        return (f"WARN: {_c.LEDGER_PATH} missing — run scripts/seed_ledger.py "
                "--yes or expect near-zero emissions until the nightly build")
    try:
        import json as _json
        with open(_c.LEDGER_PATH, "r", encoding="utf-8") as f:
            n = len((_json.load(f).get("cohorts")) or {})
        return f"{n} cohorts (floor {_c.LEDGER_FLOOR}, min_n {_c.LEDGER_MIN_N})"
    except Exception as e:
        return f"WARN: {_c.LEDGER_PATH} unreadable ({e})"


def _universe():
    """Every SYMBOLS pair must have a fresh 1h candle — delisted/halted pairs
    (the LUNA/TON class of rot) fail loudly here instead of silently skipping
    every cycle. Network check; skipped with PREFLIGHT_SKIP_NETWORK=1."""
    if os.getenv("PREFLIGHT_SKIP_NETWORK"):
        return "skipped (PREFLIGHT_SKIP_NETWORK)"
    import time as _t
    from cycle import SYMBOLS
    from utils.data_fetcher import _get_exchange
    ex = _get_exchange("binance")
    stale = []
    for sym in SYMBOLS:
        try:
            k = ex.fetch_ohlcv(sym, timeframe="1h", limit=2)
            if not k or (_t.time() * 1000 - k[-1][0]) > 2 * 3600 * 1000:
                stale.append(sym)
        except Exception:
            stale.append(sym)
    if stale:
        raise RuntimeError(f"stale/dead pairs: {stale}")
    return f"{len(SYMBOLS)} pairs fresh on Binance spot"


check("imports", _imports)
check("sklearn", _sklearn)
check("database", _db)
check("rag embedder", _embedder)
check("env vars", _env)
check("env parity", _env_parity)
check("emission ledger", _ledger)
check("membership", _membership)
check("universe", _universe)

print("\n" + ("READY" if ok else "NOT READY — fix the XX items above"))
sys.exit(0 if ok else 1)
