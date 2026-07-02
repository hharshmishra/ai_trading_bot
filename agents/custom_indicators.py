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

_NWE_W_CACHE: dict = {}


def _nwe_weight_cache(n: int, h: float):
    """(W, row_sums) for the two-sided gaussian kernel — pure function of
    (n, h); cached because the backtest calls it thousands of times."""
    key = (n, round(h, 9))
    hit = _NWE_W_CACHE.get(key)
    if hit is not None:
        return hit
    idx = np.arange(n, dtype=float)
    d = np.subtract.outer(idx, idx)
    denom = 2.0 * (h ** 2) if h > 0 else 1e-12
    W = np.exp(-(d * d) / denom)
    sw = W.sum(axis=1)
    if len(_NWE_W_CACHE) > 8:   # a few (n, h) shapes at most; guard leaks
        _NWE_W_CACHE.clear()
    _NWE_W_CACHE[key] = (W, sw)
    return W, sw


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
        # Two-sided kernel over the passed window (LuxAlgo repaint behaviour).
        # Vectorized: the old per-element double loop was O(n^2) *Python* ops,
        # far too slow for per-bar backtest replay. Same math — the weight
        # matrix W[i,j] = gauss(i-j, h) reproduces the loop exactly (parity
        # test against _nwe_repaint_reference in tests/test_backtest_harness.py).
        # W depends only on (n, h), so it is cached: per call only the matmul runs.
        x = df["close"].values.astype(float)
        n = len(x)

        W, sw = _nwe_weight_cache(n, float(h))
        nwe_out = (W @ x) / np.where(sw == 0, 1.0, sw)
        nwe_out = np.where(sw == 0, x, nwe_out)

        # Envelope: flat band from the mean absolute error over the window
        sae = float(np.mean(np.abs(x - nwe_out))) * float(mult)

        df["nwe_out"] = nwe_out
        df["nwe_mae"] = np.full(n, sae)
        df["nwe_upper"] = nwe_out + sae
        df["nwe_lower"] = nwe_out - sae

        return df


def _nwe_repaint_reference(df: pd.DataFrame, h: float = 8.0, mult: float = 3.0) -> pd.DataFrame:
    """Original O(n^2) Python-loop repaint implementation, kept ONLY as the
    ground truth for the vectorization parity test. Do not call in production."""
    src = df["close"].values.astype(float)
    n = len(src)

    nwe_out = np.zeros(n)
    nwe_mae = np.zeros(n)

    def gauss(x, h):
        return np.exp(-(x**2) / (2 * h**2))

    for i in range(n):
        sum_w = 0.0
        sum_x = 0.0
        for j in range(n):
            w = gauss(i - j, h)
            sum_x += src[j] * w
            sum_w += w
        nwe_out[i] = sum_x / sum_w if sum_w != 0 else src[i]

    sae = np.mean(np.abs(src - nwe_out)) * mult
    nwe_mae[:] = sae
    df["nwe_out"] = nwe_out
    df["nwe_mae"] = nwe_mae
    df["nwe_upper"] = nwe_out + sae
    df["nwe_lower"] = nwe_out - sae
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

def supertrend_fast(high: pd.Series, low: pd.Series, close: pd.Series,
                    length: int = 10, multiplier: float = 3.0) -> pd.DataFrame:
    """Drop-in replacement for the vendored ``pandas_ta.supertrend``.

    Identical algorithm (same RMA ATR via pandas_ta, same band-ratchet
    recursion, same column names) but the recursion runs on numpy arrays —
    the vendored version's per-row ``.iloc`` get/set loop was ~60% of the
    whole decide() call. Parity-locked by tests/test_backtest_harness.py.
    """
    import pandas_ta as ta
    m = close.size
    hl2_ = (high + low) / 2.0
    matr = multiplier * ta.atr(high, low, close, length)
    ub = (hl2_ + matr).to_numpy(dtype=float)
    lb = (hl2_ - matr).to_numpy(dtype=float)
    c = close.to_numpy(dtype=float)

    dir_ = np.ones(m, dtype=int)
    trend = np.zeros(m)
    long_ = np.full(m, np.nan)
    short = np.full(m, np.nan)

    for i in range(1, m):
        if c[i] > ub[i - 1]:
            dir_[i] = 1
        elif c[i] < lb[i - 1]:
            dir_[i] = -1
        else:
            dir_[i] = dir_[i - 1]
            if dir_[i] > 0 and lb[i] < lb[i - 1]:
                lb[i] = lb[i - 1]
            if dir_[i] < 0 and ub[i] > ub[i - 1]:
                ub[i] = ub[i - 1]
        if dir_[i] > 0:
            trend[i] = long_[i] = lb[i]
        else:
            trend[i] = short[i] = ub[i]

    p = f"_{length}_{float(multiplier)}"
    return pd.DataFrame({f"SUPERT{p}": trend, f"SUPERTd{p}": dir_,
                         f"SUPERTl{p}": long_, f"SUPERTs{p}": short},
                        index=close.index)


def donchian_breakout_signal(df: pd.DataFrame, fast: int = 20, slow: int = 55,
                             atr_period: int = 14) -> Optional[Dict[str, Any]]:
    """Donchian channel breakout on the LAST CLOSED bar (trend trigger).

    BUY when close breaks above the prior ``fast``-bar high, SELL below the
    prior ``fast``-bar low (prior window excludes the current bar — no lookahead).
    Confidence: 0.60 base, +0.15 if the ``slow`` channel broke too (major
    breakout), plus penetration depth in ATR units (capped), max 0.95.
    Returns None when no breakout.
    """
    need = max(fast, slow) + 1
    if df is None or len(df) < need:
        return None
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = float(df["close"].astype(float).iloc[-1])

    prior_fast_high = float(high.iloc[-(fast + 1):-1].max())
    prior_fast_low = float(low.iloc[-(fast + 1):-1].min())
    prior_slow_high = float(high.iloc[-(slow + 1):-1].max())
    prior_slow_low = float(low.iloc[-(slow + 1):-1].min())

    tr = pd.concat([high - low,
                    (high - df["close"].astype(float).shift(1)).abs(),
                    (low - df["close"].astype(float).shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.rolling(atr_period).mean().iloc[-1]
    atr = float(atr) if pd.notna(atr) and atr > 0 else None

    signal, depth, major = None, 0.0, False
    if close > prior_fast_high:
        signal = "buy"
        depth = close - prior_fast_high
        major = close > prior_slow_high
    elif close < prior_fast_low:
        signal = "sell"
        depth = prior_fast_low - close
        major = close < prior_slow_low
    if signal is None:
        return None

    conf = 0.60 + (0.15 if major else 0.0)
    if atr:
        conf += 0.20 * min(depth / atr, 1.0)
    return {"signal": signal, "confidence": float(np.clip(conf, 0.5, 0.95)),
            "name": "donchian_breakout"}


def squeeze_release_signal(df: pd.DataFrame, bb_len: int = 20, bb_std: float = 2.0,
                           kc_len: int = 20, kc_mult: float = 1.5,
                           release_within: int = 3) -> Optional[Dict[str, Any]]:
    """Squeeze-momentum release (TTM-squeeze style, trend trigger).

    Squeeze ON = Bollinger bands inside Keltner channel (vol compression).
    Fires when the squeeze released within the last ``release_within`` bars;
    direction = sign of close minus the ``bb_len`` midline; confidence scales
    with momentum strength vs its own recent range.
    """
    need = max(bb_len, kc_len) + release_within + 2
    if df is None or len(df) < need:
        return None
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)

    mid = close.rolling(bb_len).mean()
    sd = close.rolling(bb_len).std(ddof=0)
    bb_u, bb_l = mid + bb_std * sd, mid - bb_std * sd

    tr = pd.concat([high - low,
                    (high - close.shift(1)).abs(),
                    (low - close.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.rolling(kc_len).mean()
    kc_mid = close.rolling(kc_len).mean()
    kc_u, kc_l = kc_mid + kc_mult * atr, kc_mid - kc_mult * atr

    on = (bb_u < kc_u) & (bb_l > kc_l)
    on = on.fillna(False)
    if len(on) < release_within + 1:
        return None
    now_off = not bool(on.iloc[-1])
    was_on = bool(on.iloc[-(release_within + 1):-1].any())
    if not (now_off and was_on):
        return None

    mom = close - mid  # momentum proxy around the midline
    m = mom.iloc[-1]
    if pd.isna(m) or m == 0:
        return None
    scale = mom.abs().rolling(bb_len).max().iloc[-1]
    strength = float(min(abs(m) / scale, 1.0)) if pd.notna(scale) and scale > 0 else 0.5
    return {"signal": "buy" if m > 0 else "sell",
            "confidence": float(np.clip(0.55 + 0.35 * strength, 0.5, 0.9)),
            "name": "squeeze_release"}


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

    return df

def alpha_trend(df: pd.DataFrame, coeff: float = 1.0, period: int = 14, use_volume: bool = True):
    """
    AlphaTrend indicator (converted from PineScript).
    
    Parameters:
        df (pd.DataFrame): must contain ['open','high','low','close','volume']
        coeff (float): multiplier (default 1.0)
        period (int): lookback period (default 14)
        use_volume (bool): whether to use MFI (True) or RSI (False)
    
    Returns:
        df (pd.DataFrame): with added columns:
            - 'alpha_trend'
            - 'alpha_trend_prev'
            - 'alpha_signal' ('buy', 'sell', or None)
    """

    # Average True Range (ATR)
    df['h-l'] = df['high'] - df['low']
    df['h-c'] = abs(df['high'] - df['close'].shift())
    df['l-c'] = abs(df['low'] - df['close'].shift())
    tr = df[['h-l', 'h-c', 'l-c']].max(axis=1)
    df['atr'] = tr.rolling(period).mean()

    # Money Flow Index (MFI) if volume exists, else RSI
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    if use_volume:
        mf = typical_price * df['volume']
        positive_mf = mf.where(typical_price > typical_price.shift(), 0.0)
        negative_mf = mf.where(typical_price < typical_price.shift(), 0.0)
        mfi_ratio = positive_mf.rolling(period).sum() / negative_mf.rolling(period).sum()
        df['osc'] = 100 - (100 / (1 + mfi_ratio))
    else:
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        df['osc'] = 100 - (100 / (1 + rs))

    # AlphaTrend core calculation
    upT = df['low'] - df['atr'] * coeff
    downT = df['high'] + df['atr'] * coeff

    # Recursive ratchet — numpy arrays instead of per-row .iloc (same math,
    # ~10x faster; the loop itself is unavoidable due to the alpha[i-1]
    # dependency). NaN osc rows compare False on >=50, matching the original.
    n_rows = len(df)
    osc_a = df['osc'].to_numpy(dtype=float)
    upT_a = upT.to_numpy(dtype=float)
    downT_a = downT.to_numpy(dtype=float)
    alpha = np.zeros(n_rows)
    if n_rows:
        alpha[0] = float(df['close'].iloc[0])  # init
        prev = alpha[0]
        for i in range(1, n_rows):
            if osc_a[i] >= 50:
                prev = upT_a[i] if upT_a[i] > prev else prev
            else:
                prev = downT_a[i] if downT_a[i] < prev else prev
            alpha[i] = prev

    df['alpha_trend'] = alpha
    df['alpha_trend_prev'] = df['alpha_trend'].shift(2)

    # Signals
    df['alpha_signal'] = None
    df.loc[df['alpha_trend'] > df['alpha_trend_prev'], 'alpha_signal'] = 'buy'
    df.loc[df['alpha_trend'] < df['alpha_trend_prev'], 'alpha_signal'] = 'sell'
    
    return df