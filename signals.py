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


TREND_TRIGGER_NAMES = ("supertrend_flip", "donchian_breakout", "squeeze_release")


def pick_regime(indicator_block: Dict[str, Any]) -> Optional[str]:
    """Regime stamped by IndicatorAgent (Phase 2); None on legacy decisions."""
    try:
        r = indicator_block["raw"]["details"].get("regime")
        return str(r) if r else None
    except Exception:
        return None


def pick_vol_ok(indicator_block: Dict[str, Any]) -> bool:
    """Volume confirmation from the decision; missing info never blocks."""
    try:
        v = indicator_block["raw"]["details"]["regime_feats"].get("vol_ok")
        return True if v is None else bool(v)
    except Exception:
        return True


def pick_trigger(indicator_block: Dict[str, Any]) -> Tuple[Optional[str], set]:
    """Majority direction among fired trend triggers, and the set of trigger
    names that voted that direction. Ties -> (None, empty)."""
    try:
        direct = indicator_block["raw"]["details"]["direct_signals"]
    except Exception:
        return None, set()
    votes: Dict[str, set] = {"buy": set(), "sell": set()}
    for d in direct:
        name = str(d.get("name", "")).lower()
        sig = str(d.get("signal", "")).lower()
        if name in TREND_TRIGGER_NAMES and sig in ("buy", "sell"):
            votes[sig].add(name)
    if len(votes["buy"]) == len(votes["sell"]):
        return None, set()
    action = "buy" if len(votes["buy"]) > len(votes["sell"]) else "sell"
    return action, votes[action]


def should_emit_signal_v2(res: Dict[str, Any]) -> Tuple[bool, str, str, float, str]:
    """Regime-gated signal gate (Phase 2, behind GATE_V2_ENABLED).

    Truth table (plan A4): NWE owns ranging regimes (its mean-reversion edge),
    trend triggers own trending regimes (NWE suppressed — the 1h baseline
    measured 33% TB precision for counter-trend NWE), volume confirms every
    band/trigger entry, 1h keeps its strictness (no confidence-only emissions).
    On suppression the reason string explains why (gate funnel telemetry).
    Decisions without a regime stamp fall back to the v1 gate.
    """
    import config as _cfg

    if not res:
        return False, "skip", "skip", 0.0, ""

    ind_block = res.get("agents", {}).get("indicator", {})
    regime = pick_regime(ind_block)
    if regime is None:
        return should_emit_signal(res)

    final_action = str(res.get("final", {}).get("action", "skip")).lower()
    final_conf = float(res.get("final", {}).get("confidence", 0.0) or 0.0)
    tf = str(res.get("timeframe", "")).lower()
    nwe_action = pick_nwe_signal(ind_block) or "skip"
    nwe_hit = nwe_action in ("buy", "sell")
    conf_hit = final_conf >= _cfg.CONFIDENCE_GATE
    vol_ok = pick_vol_ok(ind_block)
    trend_action, trend_names = pick_trigger(ind_block)
    trending = regime in ("trend_up", "trend_down")
    trend_dir = "buy" if regime == "trend_up" else "sell"

    def out(emit: bool, action: str, reason: str):
        return emit, action, nwe_action, final_conf, reason

    if tf == TF_1H:
        if trending:
            if _cfg.GATE_1H_TREND and trend_action == trend_dir and trend_names and vol_ok:
                return out(True, trend_action, "trend_1h")
            return out(False, final_action,
                       "nwe_trend_suppressed" if nwe_hit else "")
        if nwe_hit:
            if not vol_ok:
                return out(False, final_action, "low_volume")
            if regime == "mixed" and final_action != nwe_action:
                return out(False, final_action, "no_brain_agreement")
            return out(True, nwe_action,
                       "nwe_mixed" if regime == "mixed" else "nwe_ranging")
        return out(False, final_action, "")

    # 4h / 1d / 1w
    if trending:
        if trend_names:
            if not vol_ok:
                return out(False, final_action, "low_volume")
            if trend_action == trend_dir:
                return out(True, trend_action, "trend_continuation")
            if "supertrend_flip" in trend_names:
                # Backtest amendment: flip-against-trend graded 0/6 decided —
                # emits only when explicitly re-enabled.
                if _cfg.GATE_TREND_REVERSAL:
                    return out(True, trend_action, "trend_reversal")
                return out(False, final_action, "reversal_disabled")
            return out(False, final_action, "counter_trend_no_flip")
        if conf_hit:
            if final_action == trend_dir:
                return out(True, final_action, "conf_over_80")
            return out(False, final_action, "counter_trend_conf")
        return out(False, final_action,
                   "nwe_trend_suppressed" if nwe_hit else "")

    # ranging / mixed
    # Backtest amendment: NWE on higher TFs graded 12.5% (4h ranging) — NWE
    # emissions stay 1h-only unless explicitly re-enabled; conf path remains.
    nwe_allowed = _cfg.GATE_NWE_HIGHER_TF
    if nwe_allowed and nwe_hit and vol_ok and (regime != "mixed" or final_action == nwe_action):
        return out(True, nwe_action,
                   "nwe_mixed" if regime == "mixed" else "nwe_ranging")
    if conf_hit:
        return out(True, final_action, "conf_over_80")
    if nwe_hit and not nwe_allowed:
        return out(False, final_action, "nwe_higher_tf_disabled")
    if nwe_hit and not vol_ok:
        return out(False, final_action, "low_volume")
    if nwe_hit:
        return out(False, final_action, "no_brain_agreement")
    return out(False, final_action, "")


_REASON_TEXT = {
    "nwe_direct": "NWE",
    "nwe_ranging": "NWE (ranging regime)",
    "nwe_mixed": "NWE + brain agreement",
    "trend_continuation": "Trend continuation",
    "trend_reversal": "SuperTrend reversal",
    "trend_1h": "Trend trigger (1h)",
    "conf_over_80": "Confidence > 80%",
}


def fmt_signal_message(pair: str, tf: str, overall_action: str, nwe_action: str,
                       conf: float, reason: str, *,
                       regime: Optional[str] = None,
                       trigger: Optional[str] = None,
                       calibrated_conf: Optional[float] = None,
                       deriv_note: Optional[str] = None) -> str:
    reason_text = _REASON_TEXT.get(reason, "NWE" if reason == "nwe_direct" else "Confidence > 80%")
    conf_pc = f"{conf * 100:.2f}%"
    extra = ""
    if regime:
        arrow = {"trend_up": "TRENDING ↑", "trend_down": "TRENDING ↓",
                 "ranging": "RANGING", "mixed": "MIXED"}.get(regime, regime.upper())
        extra += f"📐 <b>REGIME:</b> {arrow}\n"
    if trigger:
        extra += f"🎯 <b>TRIGGER:</b> {trigger}\n"
    if calibrated_conf is not None:
        extra += f"🎚 <b>CAL. CONFIDENCE:</b> {calibrated_conf * 100:.0f}%\n"
    if deriv_note:
        extra += f"🧲 <b>DERIVS:</b> {deriv_note}\n"
    if extra:
        extra = extra.rstrip("\n") + "\n"
    return (
        "<b>🚨 SIGNAL ALERT 🚨</b>\n\n"
        f"<b>OVERALL TRADE SIGNAL:</b> {overall_action.upper()}\n"
        f"<b>NWE SIGNAL:</b> {nwe_action.upper()}\n"
        f"💱 <b>PAIR:</b> {pair}\n"
        f"⏰ <b>TIMEFRAME:</b> {tf}\n"
        f"📊 <b>CONFIDENCE:</b> {conf_pc}\n"
        f"🧠 <b>REASON:</b> {reason_text}\n"
        f"{extra}\n"
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
    for name in ("indicator", "research", "news", "derivatives"):
        a = agents.get(name)
        if a is None:
            continue  # e.g. derivatives absent on legacy decisions
        lines.append(f"• {name}: {str(a.get('action','?')).upper()} (conf {a.get('confidence')})")
    meta = res.get("meta") or {}
    if meta.get("meta_p") is not None:
        lines.append(f"• meta p(correct): {round(float(meta['meta_p']), 3)}")
    nwe = pick_nwe_signal(agents.get("indicator", {}))
    if nwe:
        lines.append(f"• NWE direct: {nwe.upper()}")
    if outcome:
        lines.append(f"\nrealized: <b>{str(outcome.get('realized_label','?')).upper()}</b> "
                     f"(fwd {outcome.get('realized_return')})")
        if outcome.get("label_tb"):
            hit = outcome.get("barrier_hit_idx")
            lines.append(f"TB: <b>{str(outcome['label_tb']).upper()}</b>"
                         + (f"@{hit}" if hit else "")
                         + (f" exit {outcome.get('exit_price')}" if outcome.get("exit_price") else ""))
    return "\n".join(lines)
