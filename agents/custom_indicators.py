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
    repaint: bool = True  # True (LuxAlgo repaint variant) IS the production path —
                          # the whole backtest evidence base was measured on it.
                          # No lookahead at decision time: the window ends at the
                          # newest closed bar, and the backtest replays this exact
                          # windowed call per bar (sim/live parity by construction).
                          # repaint=False = endpoint/non-repainting variant, unused.
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


def direct_signal_from_nwee(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """EVENT-based NWE signal (used when config.NWE_EVENT_MODE is on).

    Inspect the LAST CLOSED BAR and return a discrete signal only if a band
    CROSSING occurred on that bar:
      - BUY  if crossunder(close, lower)
      - SELL if crossover(close, upper)
    Confidence scales with how far the close finished outside the band relative
    to the band width. Re-firing requires price to re-enter the band first —
    an inherent cooldown the state-based variant lacks.
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

def pivot_divergence(price: pd.Series, osc: pd.Series, pivot_w: int = 3,
                     lookback: int = 60, recency: int = 12) -> int:
    """Classic two-pivot divergence (enhancement D1/D2): +1 bullish (price
    lower-low, oscillator higher-low), -1 bearish (price higher-high,
    oscillator lower-high), 0 otherwise.

    A pivot needs ``pivot_w`` bars on EACH side, so it confirms ``pivot_w``
    bars late — no lookahead by construction. Only fires when the newest
    confirming pivot lies within the last ``recency`` bars (stale divergences
    are noise). O(n) single pass over the last ``lookback`` bars.
    """
    n = len(price)
    need = 2 * pivot_w + 1
    if n < need + 2 or len(osc) != n:
        return 0
    p = price.astype(float).to_numpy()[-lookback:]
    o = osc.astype(float).to_numpy()[-lookback:]
    m = len(p)

    # Prominence floor: a pivot must stand out from its window by a multiple
    # of the series' robust noise scale (1.4826*MAD), else every micro-wiggle
    # counts as a pivot and shadows the real swing points.
    med = float(np.nanmedian(p))
    mad = float(np.nanmedian(np.abs(p - med))) or 1e-9
    min_prom = 3.0 * 1.4826 * mad

    lows, highs = [], []          # (index, price, osc) of confirmed pivots
    for i in range(pivot_w, m - pivot_w):
        win_p = p[i - pivot_w: i + pivot_w + 1]
        if np.isnan(win_p).any() or np.isnan(o[i]):
            continue
        if p[i] == win_p.min() and (win_p > p[i]).sum() >= 2 * pivot_w - 1:
            if (win_p.max() - p[i]) >= min_prom:
                lows.append((i, p[i], o[i]))
        if p[i] == win_p.max() and (win_p < p[i]).sum() >= 2 * pivot_w - 1:
            if (p[i] - win_p.min()) >= min_prom:
                highs.append((i, p[i], o[i]))

    def _recent(pivots):
        return len(pivots) >= 2 and pivots[-1][0] >= m - pivot_w - recency

    if _recent(lows):
        (i1, p1, o1), (i2, p2, o2) = lows[-2], lows[-1]
        if p2 < p1 and o2 > o1:
            return 1
    if _recent(highs):
        (i1, p1, o1), (i2, p2, o2) = highs[-2], highs[-1]
        if p2 > p1 and o2 < o1:
            return -1
    return 0


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
    # no valid momentum range -> NO strength evidence (0.0), not "moderate";
    # only reachable on very short frames (<2*bb_len bars), never at the
    # production window of 500
    strength = float(min(abs(m) / scale, 1.0)) if pd.notna(scale) and scale > 0 else 0.0
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
    dir_val = pd.Series(dir_val).ffill()  # forward-fill direction (pandas 2.1+ dropped the old method kwarg)

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
        pos_sum = positive_mf.rolling(period).sum()
        neg_sum = negative_mf.rolling(period).sum()
        df['osc'] = 100 - (100 / (1 + pos_sum / neg_sum))
        # dead-flat window: 0/0 -> NaN, which the ratchet below reads as
        # "downtrend" (NaN >= 50 is False). A market that has not moved at
        # all is neutral, not bearish.
        df.loc[(pos_sum == 0) & (neg_sum == 0), 'osc'] = 50.0
    else:
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        df['osc'] = 100 - (100 / (1 + gain / loss))
        df.loc[(gain == 0) & (loss == 0), 'osc'] = 50.0

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
    # shift(2) is FAITHFUL to the original AlphaTrend (Pine:
    # ta.crossover(AlphaTrend, AlphaTrend[2])) — 2-bar comparison is the
    # indicator's designed flip detection, not an off-by-one.
    df['alpha_trend_prev'] = df['alpha_trend'].shift(2)

    # Signals
    df['alpha_signal'] = None
    df.loc[df['alpha_trend'] > df['alpha_trend_prev'], 'alpha_signal'] = 'buy'
    df.loc[df['alpha_trend'] < df['alpha_trend_prev'], 'alpha_signal'] = 'sell'

    return df


# ---------------------------------------------------------------------------
# v3.4 extra type-2 confluence votes (config.T2_EXTRA_VOTES)
#
# Pure functions over the CLOSED-candle frame, each returning a vote in
# {-1, 0, +1}. State-based like the existing type-2 rules (supertrend_dir,
# MA ribbon) — votes are continuous context, not entry triggers. Every vote
# ships default-OFF and must win its backtest A/B (tb_precision delta > 0,
# significant at 95%, expectancy not degraded) before default promotion.
# ---------------------------------------------------------------------------

def _simple_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                period: int = 14) -> float:
    """Simple-mean true range over the last ``period`` bars — same math as
    grading.barriers.atr_from_ohlcv, inlined to keep this module layer-free."""
    if len(close) < period + 1:
        return 0.0
    h, l, c = high[-(period + 1):], low[-(period + 1):], close[-(period + 1):]
    tr = np.maximum(h[1:] - l[1:],
                    np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
    return float(tr.mean())


def rsi30_vote(close: pd.Series) -> int:
    """Slow RSI-30 extremes. Multi-period RSI variants (RSI30 alongside RSI14)
    top the feature-importance rankings in crypto direction studies. The
    longer period compresses the range, hence 35/65 rather than 30/70."""
    try:
        import pandas_ta as ta
        r = ta.rsi(close, length=30)
        if r is None or pd.isna(r.iloc[-1]):
            return 0
        v = float(r.iloc[-1])
        return 1 if v < 35 else (-1 if v > 65 else 0)
    except Exception:
        return 0


def mfi_vote(high: pd.Series, low: pd.Series, close: pd.Series,
             volume: pd.Series) -> int:
    """Money Flow Index 14 — volume-weighted RSI. Oversold <20 / overbought >80."""
    try:
        import pandas_ta as ta
        m = ta.mfi(high, low, close, volume, length=14)
        if m is None or pd.isna(m.iloc[-1]):
            return 0
        v = float(m.iloc[-1])
        return 1 if v < 20 else (-1 if v > 80 else 0)
    except Exception:
        return 0


def cci_vote(high: pd.Series, low: pd.Series, close: pd.Series) -> int:
    """CCI-20 beyond ±100 — classic mean-reversion extreme."""
    try:
        import pandas_ta as ta
        c = ta.cci(high, low, close, length=20)
        if c is None or pd.isna(c.iloc[-1]):
            return 0
        v = float(c.iloc[-1])
        return 1 if v < -100 else (-1 if v > 100 else 0)
    except Exception:
        return 0


def vwap_vote(df: pd.DataFrame) -> int:
    """Close vs BOTH the UTC-day anchored VWAP and a rolling ~24h VWAP.

    Crypto trades 24/7, so the anchor is the UTC midnight reset (the accepted
    convention). Intraday timeframes only: on 1d+ bars a daily VWAP is just the
    bar's own typical price, so the vote abstains. Above both -> +1 (buyers own
    the session), below both -> -1, mixed -> 0.
    """
    try:
        if len(df) < 30:
            return 0
        ts = pd.to_datetime(df["timestamp"])
        sec = float(ts.diff().dt.total_seconds().median())
        if not sec or sec <= 0 or sec >= 86400:          # 1d/1w bars: abstain
            return 0
        bars_24h = max(int(round(86400 / sec)), 2)
        tp = (df["high"].astype(float) + df["low"].astype(float)
              + df["close"].astype(float)) / 3.0
        vol = df["volume"].astype(float).clip(lower=0.0)
        pv = tp * vol
        day = ts.dt.floor("D")
        anchored = pv.groupby(day).cumsum() / vol.groupby(day).cumsum().replace(0.0, np.nan)
        rolling = pv.rolling(bars_24h).sum() / vol.rolling(bars_24h).sum().replace(0.0, np.nan)
        a, r = anchored.iloc[-1], rolling.iloc[-1]
        close = float(df["close"].iloc[-1])
        if pd.isna(a) or pd.isna(r):
            return 0
        if close > float(a) and close > float(r):
            return 1
        if close < float(a) and close < float(r):
            return -1
        return 0
    except Exception:
        return 0


def ichimoku_vote(high: pd.Series, low: pd.Series, close: pd.Series) -> int:
    """Ichimoku trend agreement: close vs the CLOSED cloud + TK cross.

    Computed manually so only information available at the bar is used: the
    spans compared against today's close were derived 26 bars ago (standard
    displacement) — the pandas_ta variant's forward-projected frame is never
    touched, so there is no future leak by construction.
    """
    try:
        if len(close) < 80:                               # 52 + 26 displacement
            return 0
        h, l = high.astype(float), low.astype(float)
        tenkan = (h.rolling(9).max() + l.rolling(9).min()) / 2.0
        kijun = (h.rolling(26).max() + l.rolling(26).min()) / 2.0
        span_a = ((tenkan + kijun) / 2.0).shift(26)
        span_b = ((h.rolling(52).max() + l.rolling(52).min()) / 2.0).shift(26)
        c = float(close.iloc[-1])
        sa, sb = span_a.iloc[-1], span_b.iloc[-1]
        tk, kj = tenkan.iloc[-1], kijun.iloc[-1]
        if any(pd.isna(x) for x in (sa, sb, tk, kj)):
            return 0
        top, bot = max(float(sa), float(sb)), min(float(sa), float(sb))
        if c > top and float(tk) > float(kj):
            return 1
        if c < bot and float(tk) < float(kj):
            return -1
        return 0
    except Exception:
        return 0


# --------------------------- Fibonacci confluence --------------------------

def confirmed_swings(high: pd.Series, low: pd.Series, w: int = 3,
                     lookback: int = 120):
    """Confirmed swing pivots on high/low, same confirmation rule as
    pivot_divergence: a pivot needs ``w`` bars on EACH side, so it exists only
    ``w`` bars after the extreme — once confirmed it NEVER changes (repaint-safe
    by construction; the NWE-repaint lesson). Returns (highs, lows) as lists of
    (position, price) within the trailing ``lookback`` window."""
    h = high.astype(float).to_numpy()[-lookback:]
    l = low.astype(float).to_numpy()[-lookback:]
    m = len(h)
    highs, lows = [], []
    for i in range(w, m - w):
        win_h = h[i - w: i + w + 1]
        win_l = l[i - w: i + w + 1]
        if not (np.isnan(win_h).any() or np.isnan(win_l).any()):
            if h[i] == win_h.max() and (win_h < h[i]).sum() >= 2 * w - 1:
                highs.append((i, float(h[i])))
            if l[i] == win_l.min() and (win_l > l[i]).sum() >= 2 * w - 1:
                lows.append((i, float(l[i])))
    return highs, lows


FIB_RATIOS = (0.382, 0.5, 0.618, 0.65, 0.786)


def fib_levels(swing_lo: float, swing_hi: float) -> Dict[float, float]:
    """Retracement price at each ratio for an UP leg lo->hi (price pulls back
    DOWN from hi). For a down leg pass the same values; the caller mirrors."""
    rng = swing_hi - swing_lo
    return {r: swing_hi - r * rng for r in FIB_RATIOS}


def fib_confluence_vote(df: pd.DataFrame, w: int = 3,
                        lookback: int = 120) -> Dict[str, Any]:
    """Golden-pocket confluence vote (the TradingView fib usage, honest form).

    Evidence says fib levels carry no standalone edge, modest edge as reversal
    ZONES at 0.5-0.618 — so this only votes when price is INSIDE the golden
    pocket of the last confirmed leg (+-0.25 ATR tolerance) AND the last candle
    rejects in the pocket's direction. Returns {"vote", "ratio", "leg"}.
    """
    out = {"vote": 0, "ratio": 0.0, "leg": ""}
    try:
        if len(df) < 2 * w + 5:
            return out
        highs, lows = confirmed_swings(df["high"], df["low"], w=w, lookback=lookback)
        if not highs or not lows:
            return out
        (hi_i, hi_p), (lo_i, lo_p) = highs[-1], lows[-1]
        if hi_p <= lo_p:
            return out
        atr = _simple_atr(df["high"].astype(float).to_numpy(),
                          df["low"].astype(float).to_numpy(),
                          df["close"].astype(float).to_numpy())
        tol = 0.25 * atr
        close = float(df["close"].iloc[-1])
        opn = float(df["open"].iloc[-1])
        rng = hi_p - lo_p
        if hi_i > lo_i:
            # up leg lo->hi; pocket = prices at 0.5..0.618 retracement (below hi)
            zone_hi = hi_p - 0.5 * rng
            zone_lo = hi_p - 0.618 * rng
            if zone_lo - tol <= close <= zone_hi + tol and close > opn:
                ratio = (hi_p - close) / rng if rng > 0 else 0.0
                return {"vote": 1, "ratio": round(float(ratio), 3), "leg": "up"}
        else:
            # down leg hi->lo; pocket = 0.5..0.618 bounce zone (above lo)
            zone_lo = lo_p + 0.5 * rng
            zone_hi = lo_p + 0.618 * rng
            if zone_lo - tol <= close <= zone_hi + tol and close < opn:
                ratio = (close - lo_p) / rng if rng > 0 else 0.0
                return {"vote": -1, "ratio": round(float(ratio), 3), "leg": "down"}
        return out
    except Exception:
        return out

# ---------------------------------------------------------------------------
# Smart Money Structure (v3.8) — port of "Smart Money Structure | GainzAlgo"
# (Pine v5, operator-supplied). Three independent direct-signal sources:
#   sms       — the chart's BUY/SELL labels: vol-adaptive momentum burst
#               confirmed by volume expansion + N-bar breakout + min distance
#   sms_bos   — Break of Structure: close-colored breach of the PREVIOUS
#               confirmed pivot (continuation flavor)
#   sms_choch — Change of Character: colored cross of the CURRENT pivot level
#               (early-reversal flavor)
# Deliberate deviations from the Pine source, documented for the audit trail:
#   * the multi-timeframe EMA/VWAP trend FILTERS are not applied to the
#     signals (Pine defaults pointed them at 5-minute data — meaningless for
#     a 1h+ system); the trend matrix ships as METRICS instead
#     (sms_trend_matrix) and the v3.8 evidence ledger prices each source per
#     regime/vol cohort, which is the honest version of the same idea.
#   * `restrict_repeated_signals` (stateful, needs the 5M trend) is replaced
#     by min-distance + the crossing nature of BOS/CHoCH (same inherent
#     cooldown the NWE event mode uses).
#   * ATR is the codebase's simple rolling-mean TR, not Pine's RMA.
# All computations are causal with bounded lookback, so full-series output
# equals per-window output for bars past warmup (backtest-vectorizable, same
# argument as the NWE vectorization).
# ---------------------------------------------------------------------------

def sms_structure(df: pd.DataFrame, pivot_len: int = 5,
                  momentum_base: float = 0.01, min_dist: int = 5,
                  vol_long: int = 50, vol_short: int = 5,
                  breakout_len: int = 5) -> Optional[pd.DataFrame]:
    """Adds per-bar SMS columns; returns the df (copy) or None if too short.

    Columns: sms_last_high/sms_last_low (confirmed pivot state),
    sms_bos_buy/sms_bos_sell/sms_choch_buy/sms_choch_sell (structure events),
    sms_buy/sms_sell (filtered BUY/SELL label events).
    """
    need = max(2 * pivot_len + 2, vol_long + 1, breakout_len + 2, 16)
    if df is None or len(df) < need:
        return None
    d = df.copy()
    high = d["high"].astype(float)
    low = d["low"].astype(float)
    close = d["close"].astype(float)
    opn = d["open"].astype(float)
    vol = d["volume"].astype(float) if "volume" in d else pd.Series(0.0, index=d.index)

    # --- confirmed pivots (ta.pivothigh/low(len,len)): a STRICT local
    # extreme vs both len-bar sides, visible only len bars later. Strictness
    # (not >=) keeps flat stretches from minting a pivot on every bar. ---
    left_hi = high.rolling(pivot_len).max().shift(1)
    right_hi = high.iloc[::-1].rolling(pivot_len).max().iloc[::-1].shift(-1)
    left_lo = low.rolling(pivot_len).min().shift(1)
    right_lo = low.iloc[::-1].rolling(pivot_len).min().iloc[::-1].shift(-1)
    piv_hi = high.where((high > left_hi) & (high > right_hi))
    piv_lo = low.where((low < left_lo) & (low < right_lo))
    last_high = piv_hi.shift(pivot_len).ffill()
    last_low = piv_lo.shift(pivot_len).ffill()
    d["sms_last_high"] = last_high
    d["sms_last_low"] = last_low

    red = close < opn
    green = close > opn

    def _xunder(a: pd.Series, b: pd.Series) -> pd.Series:
        return (a.shift(1) >= b.shift(1)) & (a < b)

    def _xover(a: pd.Series, b: pd.Series) -> pd.Series:
        return (a.shift(1) <= b.shift(1)) & (a > b)

    d["sms_choch_sell"] = (_xunder(low, last_high) & red).fillna(False)
    d["sms_choch_buy"] = (_xover(high, last_low) & green).fillna(False)
    prev_ll = last_low.shift(1)
    prev_lh = last_high.shift(1)
    d["sms_bos_sell"] = (_xunder(low, prev_ll) & (low < prev_ll) & red).fillna(False)
    d["sms_bos_buy"] = (_xover(high, prev_lh) & (high > prev_lh) & green).fillna(False)

    # --- BUY/SELL label conditions: vol-adaptive momentum + filters ---
    pc = close.pct_change() * 100.0
    tr = pd.concat([high - low, (high - close.shift(1)).abs(),
                    (low - close.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    atr = atr.fillna(high - low)                    # Pine's na fallback
    vf = (atr / close).clip(lower=0.0)
    thr = momentum_base * (1.0 + 2.0 * vf)
    early_buy = pc > thr
    early_sell = pc < -thr

    vol_sma_long = vol.rolling(vol_long).mean()
    vol_sma_short = vol.rolling(vol_short).mean()
    vol_ok = (vol > vol_sma_long) & (vol_sma_short.diff() > 0)

    hi_roll = high.rolling(breakout_len).max().shift(1)
    lo_roll = low.rolling(breakout_len).min().shift(1)
    brk_buy = close > hi_roll
    brk_sell = close < lo_roll

    raw_buy = (early_buy & vol_ok & brk_buy).fillna(False).to_numpy()
    raw_sell = (early_sell & vol_ok & brk_sell).fillna(False).to_numpy()

    # min-signal-distance state machine (shared bar counter, as in Pine)
    buy_out = np.zeros(len(d), dtype=bool)
    sell_out = np.zeros(len(d), dtype=bool)
    last_bar = -min_dist - 1
    for i in range(len(d)):
        if i - last_bar < min_dist:
            continue
        if raw_sell[i]:
            sell_out[i] = True
            last_bar = i
        elif raw_buy[i]:
            buy_out[i] = True
            last_bar = i
    d["sms_buy"] = buy_out
    d["sms_sell"] = sell_out
    return d


def sms_signal_from(df: pd.DataFrame, pivot_len: int = 5,
                    momentum_base: float = 0.01, min_dist: int = 5,
                    vol_long: int = 50, vol_short: int = 5,
                    breakout_len: int = 5) -> Optional[Dict[str, Any]]:
    """Direct signal for the LAST CLOSED bar, or None when nothing fired.
    Priority when several fire on one bar: label (fullest confluence) > BOS
    > CHoCH. Confidence is a static prior; the EB direct-conf layer replaces
    it once >=30 graded fires exist (same contract as every direct signal).
    """
    d = sms_structure(df, pivot_len=pivot_len, momentum_base=momentum_base,
                      min_dist=min_dist, vol_long=vol_long,
                      vol_short=vol_short, breakout_len=breakout_len)
    if d is None:
        return None
    last = d.iloc[-1]
    if last["sms_buy"]:
        return {"signal": "buy", "confidence": 0.62, "name": "sms"}
    if last["sms_sell"]:
        return {"signal": "sell", "confidence": 0.62, "name": "sms"}
    if last["sms_bos_buy"]:
        return {"signal": "buy", "confidence": 0.60, "name": "sms_bos"}
    if last["sms_bos_sell"]:
        return {"signal": "sell", "confidence": 0.60, "name": "sms_bos"}
    if last["sms_choch_buy"]:
        return {"signal": "buy", "confidence": 0.58, "name": "sms_choch"}
    if last["sms_choch_sell"]:
        return {"signal": "sell", "confidence": 0.58, "name": "sms_choch"}
    return None


def _sms_vwap_daily(d: pd.DataFrame) -> pd.Series:
    """Session (UTC-day) anchored VWAP over hlc3 — Pine's ta.vwap analogue."""
    ts = pd.to_datetime(d["timestamp"])
    hlc3 = (d["high"].astype(float) + d["low"].astype(float)
            + d["close"].astype(float)) / 3.0
    vol = d["volume"].astype(float)
    day = ts.dt.floor("D")
    pv = (hlc3 * vol).groupby(day).cumsum()
    vv = vol.groupby(day).cumsum().replace(0.0, np.nan)
    return pv / vv


def sms_trend_matrix(fetcher, symbol: str, tfs=("1h", "4h", "1d"),
                     ema_len: int = 20, cvd_window: int = 96) -> Optional[Dict[str, Any]]:
    """EMA20 + daily-VWAP alignment per timeframe (the Pine table, remapped
    from its 1M..1D set to the system's own TFs) plus a scale-normalized CVD.

      trend[tf] = +1 close above both / -1 below both / 0 mixed
      strength  = mean(trends) * 100            (-100 .. +100)
      conf      = 90 all agree / 75 two / 60 one / 50 none  (Pine tiers)
      cvd_norm  = rolling sum(sign(dClose)*vol) / rolling sum(vol) on tfs[0]
                  (Pine's absolute 10K/50K cutoffs are symbol-scale-dependent;
                  the ratio is comparable across the whole universe)

    Live-path only (network fetches); returns None on any failure — metrics
    are context, never load-bearing.
    """
    trends: Dict[str, int] = {}
    cvd_norm = None
    try:
        for i, tf in enumerate(tfs):
            d = fetcher.get_ohlcv(symbol, tf, limit=max(ema_len * 6, cvd_window + 8))
            if d is None or len(d) < ema_len + 2:
                trends[tf] = 0
                continue
            close = d["close"].astype(float)
            ema = close.ewm(span=ema_len, adjust=False).mean()
            vwap = _sms_vwap_daily(d)
            c = float(close.iloc[-1])
            e = float(ema.iloc[-1])
            v = float(vwap.iloc[-1]) if np.isfinite(vwap.iloc[-1]) else e
            trends[tf] = 1 if (c > e and c > v) else (-1 if (c < e and c < v) else 0)
            if i == 0 and "volume" in d:
                vol = d["volume"].astype(float)
                sign = np.sign(close.diff()).fillna(0.0)
                num = (sign * vol).rolling(cvd_window).sum().iloc[-1]
                den = vol.rolling(cvd_window).sum().iloc[-1]
                if np.isfinite(num) and np.isfinite(den) and den > 0:
                    cvd_norm = float(np.clip(num / den, -1.0, 1.0))
    except Exception:
        return None
    if not trends:
        return None
    s = sum(trends.values())
    n = len(trends)
    strength = round(100.0 * s / n, 1)
    a = abs(s)
    conf = 90.0 if a == n else (75.0 if a == n - 1 else (60.0 if a >= 1 else 50.0))
    return {"trend": trends, "strength": strength, "confidence": conf,
            "cvd_norm": cvd_norm}
