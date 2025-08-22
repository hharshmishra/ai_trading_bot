# agents/custom_indicators.py
from __future__ import annotations
import math
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd

"""
Nadaraya-Watson Envelope (non-repainting port of LuxAlgo script)

Inputs (match Pine defaults):
- h (float): Bandwidth (default 8.0)
- mult (float): Envelope multiplier (default 3.0)
- src (str): which column to use ("close" by default)
- window (int): max lookback used by the kernel (default 500)
- repaint (bool): if True, Pine draws dynamic lines intra-series; for trading we keep False
                   to use the endpoint/non-repainting method.

Outputs added to df:
- 'nwe_out'   : kernel-smoothed series
- 'nwe_mae'   : SMA(|src - out|, window-1) * mult
- 'nwe_upper' : out + mae
- 'nwe_lower' : out - mae

Direct signal:
- BUY  when crossunder(close, lower) on last closed bar
- SELL when crossover(close, upper) on last closed bar
- Otherwise SKIP
"""

def _gauss_kernel(h: float, window: int) -> np.ndarray:
    # weights for lags 0..window-1  (lag 0 == current bar, causal)
    idx = np.arange(window, dtype=float)
    # exp( - (i^2) / (2*h^2) )
    denom = 2.0 * (h ** 2) if h > 0 else 1e-12
    w = np.exp(-(idx * idx) / denom)
    return w

def _causal_kernel_mean(src: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    y[t] = sum_{i=0..W-1} w[i] * src[t - i] normalized by sum of weights available
    Handles t < W by truncating the kernel to available history.
    O(N*W), but W defaults to 500 which is fine for crypto timeframes.
    """
    n = src.shape[0]
    W = w.shape[0]
    out = np.full(n, np.nan, dtype=float)
    for t in range(n):
        i0 = max(0, t - (W - 1))
        # number of points we can use
        k = t - i0 + 1  # <= W
        ww = w[:k]
        ss = src[i0:t+1]
        sw = ww.sum()
        if sw == 0:
            out[t] = np.nan
        else:
            out[t] = float(np.dot(ss[::-1], ww) / sw)  # reverse ss to align lag0 with current
    return out

def apply_nadaraya_watson_envelope(
    df: pd.DataFrame,
    h: float = 8.0,
    mult: float = 3.0,
    src: str = "close",
    window: int = 500,
    repaint: bool = True  # kept for signature parity; we implement non-repainting endpoint
) -> pd.DataFrame:
    if src not in df.columns:
        raise ValueError(f"Column '{src}' not found in DataFrame.")

    if not repaint:
        out = df.copy()
        x = out[src].astype(float).to_numpy()

        # 1) endpoint kernel mean (non-repainting)
        w = _gauss_kernel(h=float(h), window=int(window))
        y = _causal_kernel_mean(x, w)  # nwe_out

        out["nwe_out"] = y

        # 2) envelope via mae = SMA(|src - out|, window-1) * mult
        #    Pine uses length 499 when window=500
        L = max(2, window - 1)
        abs_err = np.abs(x - y)
        # SMA of length L
        mae = pd.Series(abs_err).rolling(L, min_periods=L).mean().to_numpy() * float(mult)
        out["nwe_mae"] = mae
        out["nwe_upper"] = out["nwe_out"] + out["nwe_mae"]
        out["nwe_lower"] = out["nwe_out"] - out["nwe_mae"]

        return out
    
    else:
        src = df["close"].values.astype(float)
        n = len(src)

        nwe_out = np.zeros(n)
        nwe_mae = np.zeros(n)
        nwe_upper = np.zeros(n)
        nwe_lower = np.zeros(n)
        
        def gauss(x, h):
            return np.exp(-(x**2) / (2 * h**2))
        
        for i in range(n):
                sum_w = 0.0
                sum_x = 0.0
                for j in range(n):
                    w = gauss(i - j, h)
                    sum_x += src[j] * w
                    sum_w += w
                out = sum_x / sum_w if sum_w != 0 else src[i]
                nwe_out[i] = out

        # Compute mean absolute error for the window
        sae = np.mean(np.abs(src - nwe_out))
        sae *= mult

        nwe_mae[:] = sae
        nwe_upper = nwe_out + sae
        nwe_lower = nwe_out - sae
        # Attach results to DataFrame
        df["nwe_out"] = nwe_out
        df["nwe_mae"] = nwe_mae
        df["nwe_upper"] = nwe_upper
        df["nwe_lower"] = nwe_lower
        
        return df


def _crossunder(prev_close: float, close: float, prev_thr: float, thr: float) -> bool:
    # crossunder(close, thr): previously above (or equal) then now below
    return prev_close >= prev_thr and close < thr

def _crossover(prev_close: float, close: float, prev_thr: float, thr: float) -> bool:
    # crossover(close, thr): previously below (or equal) then now above
    return prev_close <= prev_thr and close > thr

def direct_signal_from_nwe(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """
    Return signal based on last closed bar:
      - BUY  if close < nwe_lower
      - SELL if close > nwe_upper
      - else SKIP
    Confidence is scaled by how far price is outside the envelope.
    """
    cols = {"nwe_out","nwe_mae","nwe_upper","nwe_lower","close"}
    if not cols.issubset(set(df.columns)):
        return None

    d = df.dropna(subset=["nwe_upper","nwe_lower","close"])
    if len(d) < 1:
        return None

    last = d.iloc[-1]
    close = float(last["close"])
    up = float(last["nwe_upper"])
    lo = float(last["nwe_lower"])
    band = float(last["nwe_mae"])

    signal = "skip"
    conf = 0.5

    if close < lo:
        # BUY
        signal = "buy"
        if band > 1e-12:
            conf = float(np.clip(0.55 + 0.45 * abs(lo - close) / band, 0.55, 0.99))
        else:
            conf = 0.6

    elif close > up:
        # SELL
        signal = "sell"
        if band > 1e-12:
            conf = float(np.clip(0.55 + 0.45 * abs(close - up) / band, 0.55, 0.99))
        else:
            conf = 0.6

    return {"signal": signal, "confidence": conf}


def direct_signal_from_nwee(df: pd.DataFrame) -> Optional[Dict[str, Any]]: #Old without repainting
    """
    Inspect the LAST CLOSED BAR and return a discrete signal if an event occurred:
      - BUY  if crossunder(close, lower)
      - SELL if crossover(close, upper)
    Confidence is based on how far the close finished outside the band relative to the band width.
    """
    cols = {"nwe_out","nwe_mae","nwe_upper","nwe_lower","close"}
    if not cols.issubset(set(df.columns)):
        return None

    d = df.dropna(subset=["nwe_upper","nwe_lower","close"])
    if len(d) < 2:
        return None

    prev = d.iloc[-2]
    last = d.iloc[-1]

    prev_close = float(prev["close"])
    close = float(last["close"])
    prev_up = float(prev["nwe_upper"])
    prev_lo = float(prev["nwe_lower"])
    up = float(last["nwe_upper"])
    lo = float(last["nwe_lower"])

    signal = "skip"
    conf = 0.5

    if _crossunder(prev_close, close, prev_lo, lo):
        # BUY
        signal = "buy"
        # distance below lower vs envelope half-width (mae)
        band = float(last["nwe_mae"])
        if band > 1e-12:
            conf = float(np.clip(0.55 + 0.45 * abs(close - lo) / band, 0.55, 0.99))
        else:
            conf = 0.6

    elif _crossover(prev_close, close, prev_up, up):
        # SELL
        signal = "sell"
        band = float(last["nwe_mae"])
        if band > 1e-12:
            conf = float(np.clip(0.55 + 0.45 * abs(close - up) / band, 0.55, 0.99))
        else:
            conf = 0.6

    # If no crossing, we return None to let other indicators decide, or return skip with low conf.
    if signal == "skip":
        return {"signal": "skip", "confidence": 0.5}

    return {"signal": signal, "confidence": conf}

def chandelier_exit(df: pd.DataFrame, atr_period: int = 22, atr_mult: float = 3.0, use_close: bool = True):
    """
    Implements the Chandelier Exit indicator.
    
    Args:
        df (pd.DataFrame): DataFrame with ['open','high','low','close'] columns.
        atr_period (int): ATR period (default = 22).
        atr_mult (float): Multiplier for ATR (default = 3.0).
        use_close (bool): Whether to use close price for extremums (default = True).

    Returns:
        pd.DataFrame: Original df with added ['long_stop','short_stop','ce_signal'] columns.
                      ce_signal = 'buy', 'sell', or None
    """
    high = df['high']
    low = df['low']
    close = df['close']

    # --- Calculate ATR ---
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(atr_period).mean()

    atr_multiplied = atr_mult * atr

    # --- Long Stop ---
    if use_close:
        highest_close = close.rolling(atr_period).max()
        long_stop = highest_close - atr_multiplied
    else:
        highest_high = high.rolling(atr_period).max()
        long_stop = highest_high - atr_multiplied

    long_stop_prev = long_stop.shift(1)
    long_stop = np.where(close.shift(1) > long_stop_prev, np.maximum(long_stop, long_stop_prev), long_stop)

    # --- Short Stop ---
    if use_close:
        lowest_close = close.rolling(atr_period).min()
        short_stop = lowest_close + atr_multiplied
    else:
        lowest_low = low.rolling(atr_period).min()
        short_stop = lowest_low + atr_multiplied

    short_stop_prev = short_stop.shift(1)
    short_stop = np.where(close.shift(1) < short_stop_prev, np.minimum(short_stop, short_stop_prev), short_stop)

    # --- Direction ---
    dir_val = np.where(close > short_stop_prev, 1, np.where(close < long_stop_prev, -1, np.nan))
    dir_val = pd.Series(dir_val).fillna(method='ffill')  # forward-fill direction

    # --- Signals ---
    buy_signal = (dir_val == 1) & (pd.Series(dir_val).shift(1) == -1)
    sell_signal = (dir_val == -1) & (pd.Series(dir_val).shift(1) == 1)

    # --- Assign outputs ---
    df['long_stop'] = long_stop
    df['short_stop'] = short_stop
    df['ce_signal'] = None
    df.loc[buy_signal, 'ce_signal'] = 'buy'
    df.loc[sell_signal, 'ce_signal'] = 'sell'
    
    print(df)

    return df