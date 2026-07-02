"""Backtest replay engine (accuracy upgrade Phase 1).

Replays the PRODUCTION decision + gate code bar-by-bar: at bar t the engine
slices the same trailing window the live system sees (default 500 candles) and
calls the real ``IndicatorAgent.decide(ohlcv=window, log=False)`` followed by
the real gate function from ``signals``. No re-implementation — parity with
live behaviour is by construction (guarded by tests/test_backtest_harness.py).

Honest limitation (documented in every report): the news/research agents are
not backtestable (no historical LLM), so the confidence-gate path uses the
indicator-only confidence as a proxy. NWE / trend trigger paths are exact.

Every emission is labeled two ways:
  - triple-barrier (grading/barriers.py — the SAME function the live grader
    uses from Phase 3 on), and
  - legacy fixed-horizon +/-theta at k candles (for baseline comparability).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

import config
from grading.barriers import atr_from_ohlcv, barrier_prices, triple_barrier
from signals import should_emit_signal


@dataclass
class ReplayResult:
    symbol: str
    tf: str
    bars: List[Dict[str, Any]] = field(default_factory=list)
    emissions: List[Dict[str, Any]] = field(default_factory=list)
    funnel: Dict[str, int] = field(default_factory=dict)


def res_from_indicator(dec) -> Dict[str, Any]:
    """Wrap an IndicatorDecision into the gate-shaped ``res`` dict the live
    brain produces. Indicator-only: ``final`` mirrors the indicator head (the
    documented conf-path proxy)."""
    return {
        "chartName": dec.chartName,
        "timeframe": dec.timeframe,
        "final": {"action": dec.action, "confidence": dec.confidence},
        "agents": {
            "indicator": {
                "action": dec.action,
                "confidence": dec.confidence,
                "raw": {"details": dec.details},
            }
        },
    }


def _fixed_label(fwd_return: Optional[float], theta: float) -> Optional[str]:
    if fwd_return is None:
        return None
    if fwd_return >= theta:
        return "buy"
    if fwd_return <= -theta:
        return "sell"
    return "skip"


def replay_pair(
    df: pd.DataFrame,
    symbol: str,
    tf: str,
    *,
    agent,
    k: int,
    theta: float = 0.004,
    gate_fn: Callable[[Dict[str, Any]], Tuple] = should_emit_signal,
    window: int = 500,
    warmup: Optional[int] = None,
    atr_period: Optional[int] = None,
    tp_mult: Optional[float] = None,
    sl_mult: Optional[float] = None,
) -> ReplayResult:
    """Replay one pair over ``df`` (chronological OHLCV with timestamp column).

    ``k``/``theta`` are the grading horizon and fixed-horizon threshold for this
    timeframe (callers wire them from grader.HORIZON_K / grader.THRESHOLD).
    """
    warmup = warmup or window
    atr_period = atr_period or config.ATR_LEN
    if tp_mult is None or sl_mult is None:
        m = config.BARRIER_MULTS.get(tf, (1.5, 1.0))
        tp_mult = tp_mult if tp_mult is not None else m[0]
        sl_mult = sl_mult if sl_mult is not None else m[1]

    df = df.reset_index(drop=True)
    n = len(df)
    closes = df["close"].astype(float).to_numpy()
    result = ReplayResult(symbol=symbol, tf=tf)

    for t in range(max(warmup, window) - 1, n):
        # reset_index: live frames arrive 0-based from ccxt; several indicator
        # functions (chandelier) align pd.Series by positional index and break
        # on a shifted index. The engine must hand decide() the exact shape
        # live sees.
        wdf = df.iloc[t - window + 1: t + 1].reset_index(drop=True)
        dec = agent.decide(symbol, tf, ohlcv=wdf, log=False)
        res = res_from_indicator(dec)
        emit, overall, nwe, conf, reason = gate_fn(res)

        regime = (dec.details or {}).get("regime")  # None until Phase 2
        key = reason if emit else f"suppressed:{reason or 'no_trigger'}"
        result.funnel[key] = result.funnel.get(key, 0) + 1
        result.bars.append({
            "t": t, "ts": str(df["timestamp"].iloc[t]), "emit": emit,
            "action": overall if emit else None, "reason": reason,
            "conf": float(conf), "nwe": nwe, "regime": regime,
        })

        if not (emit and overall in ("buy", "sell")):
            continue

        entry = closes[t]
        atr = atr_from_ohlcv(wdf, period=atr_period)
        path = df.iloc[t + 1: t + 1 + k]

        label_tb, hit_idx, exit_price, ambiguous = "incomplete", None, None, False
        tp_price = sl_price = None
        if atr:
            tp_price, sl_price = barrier_prices(entry, atr, overall, tp_mult, sl_mult)
            if len(path) > 0:
                out = triple_barrier(path, entry, overall, tp_price, sl_price, k)
                label_tb, hit_idx = out.label_tb, out.hit_idx
                exit_price, ambiguous = out.exit_price, out.ambiguous

        fwd_return = ((closes[t + k] - entry) / entry) if (t + k) < n else None

        result.emissions.append({
            "t": t, "ts": str(df["timestamp"].iloc[t]), "pair": symbol, "tf": tf,
            "action": overall, "reason": reason, "regime": regime,
            "conf": float(conf), "entry": float(entry),
            "atr": float(atr) if atr else None,
            "tp_price": tp_price, "sl_price": sl_price,
            "tp_mult": float(tp_mult), "sl_mult": float(sl_mult),
            "label_tb": label_tb, "hit_idx": hit_idx, "exit_price": exit_price,
            "ambiguous": ambiguous,
            "fwd_return": fwd_return, "label_fixed": _fixed_label(fwd_return, theta),
        })

    return result
