#!/usr/bin/env python3
"""
main.py — Orchestrator & scheduler for your multi‑agent crypto trader

Phase 1 goals (implemented here):
  • Batch + parallel analysis across pairs & timeframes
  • Scheduling cadence: hourly / 4‑hourly / daily / weekly (Asia/Kolkata)
  • Telegram routing: send concise signals to a public channel; send rich JSON to dev channel
  • Confidence gate (>= 0.70) OR Nadaraya‑Watson Envelope (NWE) direct signal buy/sell
  • Feedback plumbing stubs for Phase 2 (inline buttons / callbacks)

Assumptions:
  • Your project tree matches the uploaded repo:
      Projectt/
        agents/{news_agent.py, indicator_agent.py, research_agent.py, custom_indicators.py}
        brain/decision_maker.py
        utils/data_fetcher.py
        logs/
  • brain.DecisionMaker.decide(symbol, timeframe) returns a dict like:
      {
        "chartName": "BTCUSDT",
        "timeframe": "1h",
        "agents": {"indicator": {...}, "research": {...}, "news": {...}},
        "final": {"action": "buy|sell|skip", "confidence": 0.91, "score": 0.123},
        "policy": {...}
      }
  • Symbols use the same format your agents expect (your demos use e.g. "ETCUSDT").
  • Environment variables (set on your host):
      TELEGRAM_BOT_TOKEN, TELEGRAM_SIGNALS_CHANNEL_ID, TELEGRAM_DEV_CHANNEL_ID
      (optional) TELEGRAM_TOOLS_CHANNEL_ID  # future: per‑agent on‑demand calls

Run:
  python main.py               # starts the perpetual scheduler (hourly)
  python main.py --run-once    # run a single batch with the timeframes due right now

Notes:
  • No external sched lib — a simple loop aligns to top of hour in Asia/Kolkata.
  • Concurrency guarded by a semaphore to play nicer with CCXT/OpenAI rate limits.
  • Phase 2 will wire interactive feedback via Telegram inline buttons & callbacks.
"""

from __future__ import annotations
import asyncio
import json
import os
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Sequence, Tuple

try:
    from zoneinfo import ZoneInfo
except Exception:
    # Python <3.9 fallback (not expected in your env)
    ZoneInfo = None  # type: ignore

# --- Project imports
sys.path.append(os.path.dirname(__file__))
from brain.decision_maker import DecisionMaker

# -------------- Configuration --------------
IST = ZoneInfo("Asia/Kolkata") if ZoneInfo else None
MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY"))
CONFIDENCE_GATE = float(os.getenv("CONFIDENCE_GATE"))

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_SIGNALS_CHANNEL_ID = os.getenv("TELEGRAM_SIGNALS_CHANNEL_ID")  # e.g., "-1001234567890"
TELEGRAM_DEV_CHANNEL_ID = os.getenv("TELEGRAM_DEV_CHANNEL_ID", "")
TELEGRAM_TOOLS_CHANNEL_ID = os.getenv("TELEGRAM_TOOLS_CHANNEL_ID", "")  # optional

# Pairs requested by you (mapped to USDT spot symbols)
# _TICKERS = [
#     "AAVE", "ADA", "ALGO", "AR", "ARB", "ATOM", "AVAX", "AXS", "BCH", "BNB", "BTC",
#     "CAKE", "COMP", "CRV", "DOGE", "DOT", "DYDX", "ENJ", "ETC", "ETH", "FET", "FIL", "FLOW",
#     "GALA", "GMT", "GRT", "ICP", "IMX", "INJ", "LINK", "LRC", "LUNA", "MANA",
#     "MKR", "NEAR", "OP", "POL", "PYTH", "RENDER", "SAND", "SHIB", "SNX", "SOL", "STORJ",
#     "THETA", "UNI", "WLD", "XRP", "BTCDOM"
# ]

_TICKERS = [
    "OP"
]

PAIRS: Tuple[str, ...] = tuple(f"{t}USDT" for t in _TICKERS)

# Timeframes we support (ccxt-compatible strings your agents already use)
TF_1H = "1h"
TF_4H = "4h"
TF_1D = "1d"
TF_1W = "1w"

# -------------- Utilities --------------

def now_ist() -> datetime:
    return datetime.now(IST) if IST else datetime.now()


def top_of_next_halfhour(dt: datetime) -> datetime:
    base = dt.replace(second=0, microsecond=0)
    if dt.minute < 30:
        target = base.replace(minute=30)
    else:
        target = base.replace(minute=0) + timedelta(hours=1)
    return target + timedelta(minutes=5)  # safety buffer



def due_timeframes(dt: datetime) -> Tuple[str, ...]:
    """Return the set of timeframes we should run *at this moment*.

    Rules from your spec:
      • Every hour: run 1h for all pairs
      • Every 4th hour: run 4h *and* 1h
      • Every day: run 1d, 4h, 1h
      • Every week: run 1w, 1d, 4h, 1h
    We'll compose these as a set to avoid duplication.
    """
    frames: List[str] = [TF_1H]

    hr = dt.hour
    weekday = dt.weekday()  # Monday=0 .. Sunday=6

    # 4-hourly
    if hr % 4 == 0:
        frames.append(TF_4H)

    # Daily at 00:00 local
    if hr == 0:
        frames.extend([TF_4H, TF_1D])

    # Weekly: Monday 00:00 local (adjust if you prefer Sunday)
    if weekday == 0 and hr == 0:
        frames.append(TF_1W)

    # Deduplicate in desired order of descending breadth
    order = {TF_1W: 0, TF_1D: 1, TF_4H: 2, TF_1H: 3}
    unique = sorted(set(frames), key=lambda x: order.get(x, 99))
    return tuple(unique)


def flatten_dataclass_or_dict(obj: Any) -> Dict[str, Any]:
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, dict):
        return obj
    # Try to convert common custom objects
    d = getattr(obj, "__dict__", None)
    return dict(d) if isinstance(d, dict) else {"value": str(obj)}

def format_signal_message(signal: dict) -> str:
    pair = signal.get("chartName")
    tf = signal.get("timeFrame")
    action = signal.get("action").upper()
    conf = signal.get("confidence")
    reason = signal.get("reason")
    

    msg = f"""
    <b>🚨 SIGNAL ALERT 🚨</b>\n\n
    {'📈' if action == 'BUY' else '📉' if action == 'SELL' else '⏭️'} <b>TRADE:</b> {action}\n
    💱 <b>PAIR:</b> {pair}\n
    ⏰ <b>TIMEFRAME:</b> {tf}\n
    📊 <b>CONFIDENCE:</b> {conf}\n
    🧠 <b>REASON:</b> {"NWE" if (reason == 'nwe_direct') else "Confidence > 80%"}\n\n
    ⚠️ <i>Disclaimer: This is NOT financial advice.\n
    Trading involves risk — do your own research.\n
    Sharing or reselling these signals is illegal.</i>\n\n
    ~ <b>BitReinforceX</b>\n  
    "Reinforcing your trades with AI power"
    """
    return msg.strip()





def extract_nwe_signal(agent_result: Dict[str, Any]) -> Tuple[str | None, float | None]:
    """Return (signal, confidence) from IndicatorAgent's NWE direct signal if present."""
    try:
        details = agent_result.get("raw").get("details") or {}
        ds = details.get("direct_signals")
        if not ds:
            return None, None
        # direct_signals may be a list of {name, signal, confidence} or a dict keyed by name
        if isinstance(ds, dict):
            nde = ds.get("nwe") or ds.get("NWE")
            if isinstance(nde, dict):
                sig = (nde.get("signal") or nde.get("action") or "").lower() or None
                conf = nde.get("confidence")
                return sig, float(conf) if conf is not None else None
        elif isinstance(ds, list):
            for item in ds:
                if not isinstance(item, dict):
                    continue
                nm = str(item.get("name", "")).lower()
                if nm == "nwe":
                    sig = (item.get("signal") or item.get("action") or "").lower() or None
                    conf = item.get("confidence")
                    return sig, float(conf) if conf is not None else None
    except Exception:
        pass
    return None, None


def should_route_signal(decision: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    """Apply routing policy: confidence >= gate OR NWE direct signal fires."""

    final = decision.get("final") or {}
    conf = float(final.get("confidence") or 0.0)
    reason: Dict[str, Any] = {"confidence": conf, "gate": CONFIDENCE_GATE}

    if conf >= CONFIDENCE_GATE:
        reason["rule"] = "conf_gate"
        return True, reason
    
    ind = decision.get("agents", {}).get("indicator", {})
    sig, sconf = extract_nwe_signal(ind)
    print( sig, " sig, ", sconf)
    if sig in {"buy", "sell"}:
        reason["rule"] = "nwe_direct"
        reason["nwe_signal"] = sig
        reason["nwe_confidence"] = sconf
        return True, reason
    
    reason["rule"] = "none"
    return False, reason


# -------------- Telegram --------------

def _tg_post(method: str, payload: Dict[str, Any]) -> None:
    """Fire-and-forget Telegram POST using stdlib (to avoid extra deps)."""
    if not TELEGRAM_BOT_TOKEN:
        return
    import urllib.request
    import urllib.parse

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/{method}"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as _:
            pass
    except Exception:
        # Intentionally swallow to avoid crashing the batch; logs could be added
        pass


def send_signal_message(decision: Dict[str, Any], reason: Dict[str, Any]) -> None:
    if not TELEGRAM_SIGNALS_CHANNEL_ID:
        return
    f = decision.get("final", {})
    action = (f.get("action") or "").upper()
    conf = f.get("confidence")
    sym = decision.get("chartName")
    tf = decision.get("timeframe")
    conf = conf * 100
    
    if reason.get('rule') == 'nwe_direct':
        action = reason.get('nwe_signal')
    # txt = (
    #     f"{sym} | {tf} | <b>{action}</b> (conf: {conf:.2%})\n"
    #     f"Reason: {reason.get('rule')}\n"
    # )
    txt = {
        "chartName": sym,
        "timeFrame": tf,
        "action": action,
        "confidence": f"{conf:.2f}%",
        "reason": reason.get('rule')
    }
    
    _tg_post("sendMessage", {
        "chat_id": TELEGRAM_SIGNALS_CHANNEL_ID,
        "text": format_signal_message(txt),
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    })


def send_dev_message(decision: Dict[str, Any], reason: Dict[str, Any]) -> None:
    if not TELEGRAM_DEV_CHANNEL_ID:
        return

    # Compact JSON for devs (but not huge)
    compact = {
        k: decision[k] for k in ("chartName", "timeframe", "final", "policy") if k in decision
    }
    compact["agents"] = {
        k: {
            kk: vv for kk, vv in v.items() if kk in ("action", "confidence", "details")
        } for k, v in (decision.get("agents") or {}).items()
    }

    txt = (
        f"<b>Decision</b> — {decision.get('chartName')} {decision.get('timeframe')}\n"
        f"<b>Final:</b> {decision.get('final', {}).get('action')} ({decision.get('final', {}).get('confidence'):.2%})\n"
        f"<b>Reason:</b> {json.dumps(reason)}\n"
        f"<b>JSON:</b> <code>{json.dumps(compact, ensure_ascii=False)}</code>\n\n"
        f"Reply with: /fb {decision.get('chartName')} {decision.get('timeframe')} <true_action> <news_reward>\n"
        f"Example: /fb BTCUSDT 1h buy +1"
    )
    _tg_post("sendMessage", {
        "chat_id": TELEGRAM_DEV_CHANNEL_ID,
        "text": txt,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    })


# -------------- Core batch logic --------------

async def analyze_one(brain: DecisionMaker, symbol: str, timeframe: str) -> Dict[str, Any]:
    """Run the brain once for (symbol, timeframe) in a worker thread."""
    result: Dict[str, Any] = await asyncio.to_thread(brain.decide, symbol, timeframe)
    # Ensure minimal fields are present
    result.setdefault("chartName", symbol)
    result.setdefault("timeframe", timeframe)
    return result


async def run_batch(timeframes: Sequence[str], pairs: Sequence[str] = PAIRS) -> List[Dict[str, Any]]:
    brain = DecisionMaker(prefer_csv=False)
    sem = asyncio.Semaphore(MAX_CONCURRENCY)

    async def guarded(symbol: str, tf: str) -> Dict[str, Any]:
        async with sem:
            dec = await analyze_one(brain, symbol, tf)
            ok, reason = should_route_signal(dec)
            # Log
            _log_decision({**dec, "routed": ok, "route_reason": reason, "ts": now_ist().isoformat()})
            if ok:
                send_signal_message(dec, reason)
                send_dev_message(dec, reason)
            return dec

    tasks: List[asyncio.Task] = []
    for tf in timeframes:
        for sym in pairs:
            tasks.append(asyncio.create_task(guarded(sym, tf)))

    results = await asyncio.gather(*tasks, return_exceptions=False)
    return results  # for future use (e.g., persistence/UI)


# -------------- Logging --------------

_LOG_PATH = os.path.join(os.path.dirname(__file__), "logs", "brain_predictions.jsonl")
os.makedirs(os.path.dirname(_LOG_PATH), exist_ok=True)

def _log_decision(row: Dict[str, Any]) -> None:
    try:
        with open(_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except Exception:
        pass


# -------------- Scheduler loop --------------

async def scheduler_loop(run_once: bool = False) -> None:
    while True:
        dt = now_ist()
        # frames = due_timeframes(dt)
        frames = ["1h"]
        print(f"[{dt.isoformat()}] Running frames: {frames} for {len(PAIRS)} pairs …", flush=True)
        try:
            await run_batch(frames)
        except Exception as e:
            print("Batch error:", e, file=sys.stderr)

        if run_once:
            return

        # Sleep until the top of the next hour in IST
        nxt = top_of_next_halfhour(dt)
        # guard if batch ran long: recompute from current time
        now2 = now_ist()
        wait = (top_of_next_halfhour(now2) - now2).total_seconds()
        await asyncio.sleep(max(5.0, wait))


# -------------- CLI entrypoint --------------

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-once", action="store_true", help="run a single batch then exit")
    args = ap.parse_args()

    try:
        asyncio.run(scheduler_loop(run_once=args.run_once))
    except KeyboardInterrupt:
        print("\nStopped.")
