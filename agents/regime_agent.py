"""Regime classifier (accuracy upgrade Phase 2).

Answers ONE question per (pair, timeframe): is this chart trending or ranging
right now? The answer gates which trigger family may emit signals — the NWE
band (mean-reversion) fires counter-trend all the way down a strong trend
("band walk", measured at 33% TB precision on the 1h baseline), so it is only
allowed in ranging conditions; trend triggers own trending conditions.

Deliberately NOT a brain voter: regime is non-directional, and the brain's
linear score cannot express a conditional rule — the gate can (decision A2 in
the plan). Deterministic pure function of the dataframe ⇒ thread-safe under
the fan-out, restart-safe, and bit-identical between live and backtest.

Classification (thresholds in config, hysteresis against flapping):
  enter trending : ADX >= REGIME_ADX_ENTER  OR  CHOP <= REGIME_CHOP_ENTER
  exit trending  : ADX <= REGIME_ADX_EXIT   AND CHOP >= REGIME_CHOP_EXIT
  min dwell      : REGIME_MIN_DWELL bars between flips
  direction      : +DI vs -DI (DMI) at the last bar
  ranging vs mixed (when not trending): clean range zone (exit condition
  true) -> "ranging", in-between readings -> "mixed".
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import pandas_ta as ta

import config
from grading.barriers import atr_from_ohlcv


@dataclass(frozen=True)
class RegimeSnapshot:
    regime: str            # "trend_up" | "trend_down" | "ranging" | "mixed"
    adx: Optional[float]
    chop: Optional[float]
    vol_pct: Optional[float]   # realized-vol percentile in [0, 1]
    atr: Optional[float]
    plus_di: Optional[float]
    minus_di: Optional[float]
    flips_in_walk: int

    def feats(self) -> Dict[str, Any]:
        """JSON-serializable feature dict persisted on prediction rows."""
        return {
            "regime": self.regime, "adx": self.adx, "chop": self.chop,
            "vol_pct": self.vol_pct, "atr": self.atr,
            "plus_di": self.plus_di, "minus_di": self.minus_di,
            "flips_in_walk": self.flips_in_walk,
        }


def _mixed(atr: Optional[float] = None) -> RegimeSnapshot:
    return RegimeSnapshot("mixed", None, None, None, atr, None, None, 0)


def classify_regime(df: pd.DataFrame,
                    adx_len: Optional[int] = None,
                    chop_len: Optional[int] = None,
                    walk_bars: Optional[int] = None) -> RegimeSnapshot:
    """Classify the CURRENT regime from a chronological OHLCV frame.

    Stateless hysteresis: a small state machine is walked forward over the last
    ``walk_bars`` bars of the ADX/CHOP series, so the terminal state carries
    history without any cross-call mutable state.
    """
    adx_len = adx_len or config.REGIME_ADX_LEN
    chop_len = chop_len or config.REGIME_CHOP_LEN
    walk_bars = walk_bars or config.REGIME_WALK_BARS

    if df is None or len(df) < max(adx_len, chop_len) * 3:
        return _mixed(atr_from_ohlcv(df, config.ATR_LEN) if df is not None else None)

    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)

    try:
        adx_df = ta.adx(high, low, close, length=adx_len)
        chop_s = ta.chop(high, low, close, length=chop_len)
    except Exception:
        return _mixed(atr_from_ohlcv(df, config.ATR_LEN))
    if adx_df is None or chop_s is None:
        return _mixed(atr_from_ohlcv(df, config.ATR_LEN))

    adx = adx_df[f"ADX_{adx_len}"]
    plus_di = adx_df[f"DMP_{adx_len}"]
    minus_di = adx_df[f"DMN_{adx_len}"]
    chop = chop_s if isinstance(chop_s, pd.Series) else chop_s.iloc[:, 0]

    valid = adx.notna() & chop.notna()
    if valid.sum() < config.REGIME_MIN_DWELL + 2:
        return _mixed(atr_from_ohlcv(df, config.ATR_LEN))
    adx_v = adx[valid].to_numpy()
    chop_v = chop[valid].to_numpy()

    walk = min(walk_bars, len(adx_v))
    a_w, c_w = adx_v[-walk:], chop_v[-walk:]

    in_trend = bool(a_w[0] >= config.REGIME_ADX_ENTER or c_w[0] <= config.REGIME_CHOP_ENTER)
    dwell, flips = 0, 0
    for i in range(1, walk):
        dwell += 1
        if in_trend:
            exit_ = a_w[i] <= config.REGIME_ADX_EXIT and c_w[i] >= config.REGIME_CHOP_EXIT
            if exit_ and dwell >= config.REGIME_MIN_DWELL:
                in_trend, dwell, flips = False, 0, flips + 1
        else:
            enter = a_w[i] >= config.REGIME_ADX_ENTER or c_w[i] <= config.REGIME_CHOP_ENTER
            if enter and dwell >= config.REGIME_MIN_DWELL:
                in_trend, dwell, flips = True, 0, flips + 1

    # realized-vol percentile: rolling std of log returns ranked over lookback
    logret = np.log(close / close.shift(1))
    rvol = logret.rolling(20).std()
    rv = rvol.dropna()
    vol_pct = None
    if len(rv) >= 5:
        tail = rv.iloc[-config.REGIME_VOL_LOOKBACK:]
        vol_pct = float((tail <= rv.iloc[-1]).mean())

    last_adx = float(adx_v[-1])
    last_chop = float(chop_v[-1])
    last_pdi = float(plus_di[valid].iloc[-1])
    last_mdi = float(minus_di[valid].iloc[-1])
    atr = atr_from_ohlcv(df, config.ATR_LEN)

    if in_trend:
        regime = "trend_up" if last_pdi >= last_mdi else "trend_down"
    elif (last_adx <= config.REGIME_ADX_EXIT and last_chop >= config.REGIME_CHOP_EXIT):
        regime = "ranging"
    else:
        regime = "mixed"

    return RegimeSnapshot(regime, last_adx, last_chop, vol_pct, atr,
                          last_pdi, last_mdi, flips)


class RegimeAgent:
    """Thin fetch-and-classify wrapper for the control bot and tests. The hot
    path (IndicatorAgent.decide) calls ``classify_regime`` directly on the df
    it already holds."""

    def __init__(self, prefer_csv: bool = False):
        from utils.data_fetcher import DataFetcher
        self.data = DataFetcher(prefer_csv=prefer_csv)

    def decide(self, symbol: str, timeframe: str, limit: int = 500) -> Dict[str, Any]:
        df = self.data.get_ohlcv(symbol, timeframe, limit=limit)
        snap = classify_regime(df)
        return {"agent": "regime_agent", "chartName": symbol,
                "timeframe": timeframe, **snap.feats()}
