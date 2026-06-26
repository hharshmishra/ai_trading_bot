"""Signal gating, scheduling cascade, and message formatting (Phase 4, pure).

Kept dependency-free (no telegram, no asyncio) so the rules are unit-testable.
Ported from the original main.py with the scheduler 1h-cascade bug fixed (#8).
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional, Tuple

TF_1H, TF_4H, TF_1D, TF_1W = "1h", "4h", "1d", "1w"

# seconds per timeframe — used to compute grade-due times
TF_SECONDS = {TF_1H: 3600, TF_4H: 14400, TF_1D: 86400, TF_1W: 604800}

CONFIDENCE_GATE = 0.80


def is_candle_close_minute(dt: datetime) -> bool:
    """India candle close: every :30 of the hour (Binance candles are UTC, but
    the operator schedules on IST :30)."""
    return dt.minute == 30


def timeframes_due(dt: datetime) -> list[str]:
    """Which timeframes to analyse at this candle close.

    FIX (#8): the original never scheduled 1h. Cascade:
      hourly      -> [1h]
      every 4h    -> [4h, 1h]
      daily 00:30 -> [1d, 4h, 1h]
      Monday      -> + [1w]
    """
    due: list[str] = []
    if dt.hour == 0 and dt.weekday() == 0:
        due.append(TF_1W)
    if dt.hour == 0:
        due.append(TF_1D)
    if dt.hour % 4 == 0:
        due.append(TF_4H)
    due.append(TF_1H)
    return due


def pick_nwe_signal(indicator_block: Dict[str, Any]) -> Optional[str]:
    """Extract the Nadaraya-Watson Envelope direct signal from a brain decision's
    indicator agent block (agents['indicator'])."""
    try:
        direct = indicator_block["raw"]["details"]["direct_signals"]
        for d in direct:
            if str(d.get("name", "")).lower() == "nwe":
                return str(d.get("signal", "skip")).lower()
    except Exception:
        return None
    return None


def should_emit_signal(res: Dict[str, Any]) -> Tuple[bool, str, str, float, str]:
    """Apply the operator's gating rules. Returns
    (emit, overall_action, nwe_action, confidence, reason).

    - 1h: emit ONLY on a direct NWE buy/sell (even if confidence > 80%).
    - other TFs: emit if confidence >= 0.80 OR NWE direct.
    - if both fire and disagree (e.g. conf says BUY but NWE says SELL), NWE wins.
    """
    if not res:
        return False, "skip", "skip", 0.0, ""

    final_action = str(res.get("final", {}).get("action", "skip")).lower()
    final_conf = float(res.get("final", {}).get("confidence", 0.0) or 0.0)
    tf = str(res.get("timeframe", "")).lower()

    nwe_action = pick_nwe_signal(res.get("agents", {}).get("indicator", {})) or "skip"
    nwe_hit = nwe_action in ("buy", "sell")
    conf_hit = final_conf >= CONFIDENCE_GATE

    if tf == TF_1H:
        if nwe_hit:
            return True, nwe_action, nwe_action, final_conf, "nwe_direct"
        return False, final_action, nwe_action, final_conf, ""

    if conf_hit and nwe_hit:
        return True, nwe_action, nwe_action, final_conf, "nwe_direct"   # NWE overrides
    if nwe_hit:
        return True, nwe_action, nwe_action, final_conf, "nwe_direct"
    if conf_hit:
        return True, final_action, nwe_action, final_conf, "conf_over_80"
    return False, final_action, nwe_action, final_conf, ""


def fmt_signal_message(pair: str, tf: str, overall_action: str, nwe_action: str,
                       conf: float, reason: str) -> str:
    reason_text = "NWE" if reason == "nwe_direct" else "Confidence > 80%"
    conf_pc = f"{conf * 100:.2f}%"
    return (
        "<b>🚨 SIGNAL ALERT 🚨</b>\n\n"
        f"<b>OVERALL TRADE SIGNAL:</b> {overall_action.upper()}\n"
        f"<b>NWE SIGNAL:</b> {nwe_action.upper()}\n"
        f"💱 <b>PAIR:</b> {pair}\n"
        f"⏰ <b>TIMEFRAME:</b> {tf}\n"
        f"📊 <b>CONFIDENCE:</b> {conf_pc}\n"
        f"🧠 <b>REASON:</b> {reason_text}\n\n"
        "⚠️ <i>Disclaimer: This is NOT financial advice.\n"
        "Trading involves risk – do your own research.\n"
        "Sharing or reselling these signals is illegal.</i>\n\n"
        "~ <b>BitReinforceX</b>\n  \"Reinforcing your trades with AI power\""
    )


def fmt_brain_dump(res: Dict[str, Any], outcome: Optional[Dict[str, Any]] = None) -> str:
    """Verbose per-agent breakdown for the dev channel (shown on button click)."""
    agents = res.get("agents", {})
    final = res.get("final", {})
    lines = [f"<b>🧠 BRAIN DUMP — {res.get('chartName')} {res.get('timeframe')}</b>",
             f"final: <b>{final.get('action','?').upper()}</b> conf={final.get('confidence')} score={final.get('score')}"]
    for name in ("indicator", "research", "news"):
        a = agents.get(name, {})
        lines.append(f"• {name}: {str(a.get('action','?')).upper()} (conf {a.get('confidence')})")
    nwe = pick_nwe_signal(agents.get("indicator", {}))
    if nwe:
        lines.append(f"• NWE direct: {nwe.upper()}")
    if outcome:
        lines.append(f"\nrealized: <b>{str(outcome.get('realized_label','?')).upper()}</b> "
                     f"(fwd {outcome.get('realized_return')})")
    return "\n".join(lines)
