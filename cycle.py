"""Per-cycle orchestration (Phase 4).

One analysis cycle: for each due timeframe, build the shared market context ONCE
(Phase 1), analyse every pair concurrently (concurrency-capped), record each
prediction with its grading timing (Phase 2), gate it (signals), and broadcast
emitted signals (Telegram). The grader (Phase 3) closes the loop later.
"""
from __future__ import annotations

import asyncio
import time
from typing import Any, Awaitable, Callable, Dict, List, Optional

import pandas as pd

from grader import HORIZON_K
from market_context import build_market_context
from persistence import get_store
from signals import TF_SECONDS, should_emit_signal

# Default pair universe (48 USDT pairs).
SYMBOLS = [s + "USDT" for s in [
    "AAVE", "ADA", "ALGO", "AR", "ARB", "ATOM", "AVAX", "AXS", "BCH", "BNB",
    "BTC", "CAKE", "COMP", "CRV", "DOGE", "DOT", "DYDX", "ENJ", "ETC", "ETH",
    "FET", "FIL", "FLOW", "GALA", "GMT", "GRT", "ICP", "IMX", "INJ", "LINK",
    "LRC", "LUNA", "MANA", "MKR", "NEAR", "OP", "POL", "PYTH", "RENDER", "SAND",
    "SHIB", "SNX", "SOL", "STORJ", "THETA", "UNI", "WLD", "XRP"]]


def _entry_from_df(df) -> tuple[Optional[float], Optional[int]]:
    """(entry_price, candle_close_ts_epoch) from the latest closed candle."""
    if df is None or getattr(df, "empty", True) or "timestamp" not in df.columns:
        return None, None
    last = df.iloc[-1]
    ts = pd.Timestamp(pd.to_datetime(last["timestamp"]))
    epoch = int((ts - pd.Timestamp("1970-01-01")) // pd.Timedelta("1s"))
    return float(last["close"]), epoch


async def run_cycle(
    timeframes: List[str],
    *,
    dm,
    data_fetcher,
    broadcast: Optional[Callable[..., Awaitable[Optional[str]]]] = None,
    symbols: Optional[List[str]] = None,
    store=None,
    concurrency: int = 5,
    cycle_id: Optional[str] = None,
    now_ts: Optional[float] = None,
    build_context: Callable = build_market_context,
) -> Dict[str, Any]:
    """Run one analysis cycle over ``timeframes``. Returns a summary dict."""
    store = store or get_store()
    symbols = symbols or SYMBOLS
    cycle_id = cycle_id or f"cyc-{int(now_ts or time.time())}"
    sem = asyncio.Semaphore(concurrency)
    summary = {"cycle_id": cycle_id, "analyzed": 0, "emitted": 0, "errors": 0}

    for tf in timeframes:
        # Phase 1: build the shared market context once for this timeframe.
        try:
            ctx = await asyncio.to_thread(build_context, tf, symbols, dm.indicator, dm.news, dm.research)
        except Exception:
            ctx = None

        async def analyze(sym: str, tf=tf, ctx=ctx):
            async with sem:
                try:
                    res = await asyncio.to_thread(
                        dm.decide, sym, tf, ("indicator", "research", "news"), ctx)
                except Exception:
                    summary["errors"] += 1
                    return
                summary["analyzed"] += 1

                try:
                    df = await asyncio.to_thread(data_fetcher.get_ohlcv, sym, tf, 500)
                except Exception:
                    df = None
                entry, close_ts = _entry_from_df(df)
                k = HORIZON_K.get(tf, 1)
                grade_due = (close_ts + k * TF_SECONDS.get(tf, 3600)) if close_ts else None

                emit, overall, nwe, conf, reason = should_emit_signal(res)
                session_id = None
                if emit and broadcast is not None:
                    session_id = await broadcast(pair=sym, tf=tf, overall=overall, nwe=nwe,
                                                 conf=conf, reason=reason, decision=res)
                    summary["emitted"] += 1

                store.record_prediction(
                    res, cycle_id=cycle_id, candle_close_ts=close_ts, entry_price=entry,
                    horizon_k=k, grade_due_ts=grade_due, emitted=emit, session_id=session_id)

        await asyncio.gather(*(analyze(s) for s in symbols))

    return summary
