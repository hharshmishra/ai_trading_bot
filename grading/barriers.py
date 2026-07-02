"""Triple-barrier labeling (accuracy upgrade, shared by grader AND backtest).

A directional prediction is graded by the FIRST barrier the price path touches
after the signal candle closes:

  take-profit  = entry +/- tp_mult * ATR   -> label "tp"      (direction correct)
  stop-loss    = entry -/+ sl_mult * ATR   -> label "sl"      (direction wrong)
  time barrier = k candles elapse          -> label "timeout" (no decisive move)

This is path-aware (a TP-then-crash sequence is judged by what happened first)
and volatility-adaptive (barrier width scales with ATR), unlike the legacy
fixed-horizon +/-theta labeling — which stays recorded alongside for
comparability. Living here (not in grader.py) so the backtest engine labels
emissions with the *identical* function: no sim/live labeling drift.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class BarrierOutcome:
    label_tb: str                 # "tp" | "sl" | "timeout" | "incomplete"
    hit_idx: Optional[int]        # 1-based candle index after entry (None unless tp/sl)
    exit_price: Optional[float]   # barrier price on tp/sl, k-th close on timeout
    ambiguous: bool               # both barriers inside one candle (resolved SL-first)


def barrier_prices(entry: float, atr: float, direction: str,
                   tp_mult: float, sl_mult: float) -> Tuple[float, float]:
    """(tp_price, sl_price) for a directional signal. ATR must be > 0."""
    if direction == "buy":
        return entry + tp_mult * atr, entry - sl_mult * atr
    if direction == "sell":
        return entry - tp_mult * atr, entry + sl_mult * atr
    raise ValueError(f"direction must be buy/sell, got {direction!r}")


def triple_barrier(path_df: pd.DataFrame, entry: float, direction: str,
                   tp_price: float, sl_price: float, k: int) -> BarrierOutcome:
    """Label a directional prediction from the OHLC path AFTER the entry candle.

    ``path_df`` must hold the candles strictly after ``candle_close_ts`` in
    chronological order with high/low/close columns. Scans at most ``k``
    candles. A candle whose range spans BOTH barriers is unresolvable from
    OHLC alone — we take the pessimistic SL-first reading (``ambiguous=True``),
    consistent with the system's precision-first brand.

    If no barrier is touched and fewer than ``k`` candles are available the
    outcome is "incomplete" (caller decides: grader retries later, backtest
    drops tail emissions).
    """
    if path_df is None or len(path_df) == 0:
        raise ValueError("triple_barrier needs at least one path candle")
    if direction not in ("buy", "sell"):
        raise ValueError(f"direction must be buy/sell, got {direction!r}")

    scan = path_df.iloc[: int(k)]
    highs = scan["high"].astype(float).to_numpy()
    lows = scan["low"].astype(float).to_numpy()

    for i in range(len(scan)):
        if direction == "buy":
            tp_hit = highs[i] >= tp_price
            sl_hit = lows[i] <= sl_price
        else:
            tp_hit = lows[i] <= tp_price
            sl_hit = highs[i] >= sl_price
        if tp_hit and sl_hit:
            return BarrierOutcome("sl", i + 1, float(sl_price), True)
        if sl_hit:
            return BarrierOutcome("sl", i + 1, float(sl_price), False)
        if tp_hit:
            return BarrierOutcome("tp", i + 1, float(tp_price), False)

    if len(scan) >= int(k):
        exit_price = float(scan["close"].astype(float).iloc[int(k) - 1])
        return BarrierOutcome("timeout", None, exit_price, False)
    return BarrierOutcome("incomplete", None, None, False)


def atr_from_ohlcv(df: pd.DataFrame, period: int = 14) -> Optional[float]:
    """Last-bar ATR (Wilder-free simple rolling mean of true range — matches the
    repo's chandelier/alpha_trend ATR convention). None if not computable."""
    if df is None or len(df) < period + 1:
        return None
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    tr = pd.concat([high - low,
                    (high - close.shift(1)).abs(),
                    (low - close.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.rolling(period).mean().iloc[-1]
    if pd.isna(atr) or atr <= 0:
        return None
    return float(atr)
