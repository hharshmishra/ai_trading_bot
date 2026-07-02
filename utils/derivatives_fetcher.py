"""Binance USDM futures market data (accuracy upgrade Phase 4).

Public, keyless endpoints via ccxt ``binanceusdm``: funding rate history, open
interest history, top-trader long/short ratios. All values feed the
DerivativesAgent's feature vector — positioning extremes (crowded longs paying
heavy funding into rising OI) precede squeezes at 1h–4h horizons.

Budget: one snapshot per pair per cycle, 3 REST calls each, cached for
DERIV_TTL_SECONDS (55 min) so the 4h/1d cascade reuses the hourly fetch.
48 pairs × 3 ≈ 144 req/hour against Binance's ~500/5min window. Not every
spot pair has a USDM future — ``has_futures`` gates, callers skip-vote.
"""
from __future__ import annotations

import math
import threading
import time
from typing import Any, Dict, List, Optional

import config

_LOCK = threading.Lock()        # client creation
_FETCH_LOCK = threading.Lock()  # serializes ccxt calls (sync client is not thread-safe)
_EX: Optional[Any] = None
_MARKETS: set = set()
_MARKETS_TS: float = 0.0
_CACHE: Dict[str, tuple] = {}   # spot symbol -> (fetched_ts, snapshot dict)

_MARKETS_TTL = 24 * 3600


def _client():
    global _EX
    with _LOCK:
        if _EX is None:
            import ccxt
            _EX = ccxt.binanceusdm({"enableRateLimit": True})
        return _EX


def _unified(symbol: str) -> str:
    """BTCUSDT -> BTC/USDT:USDT (ccxt unified USDM perp symbol)."""
    base = symbol[:-4] if symbol.upper().endswith("USDT") else symbol
    return f"{base}/USDT:USDT"


def _load_markets() -> set:
    """Double-checked locking: the cycle fans decide() across threads, and two
    concurrent first-call load_markets() on the shared ccxt client race inside
    ccxt — one thread loads while the rest wait."""
    global _MARKETS, _MARKETS_TS
    now = time.time()
    if _MARKETS and (now - _MARKETS_TS) < _MARKETS_TTL:
        return _MARKETS
    with _FETCH_LOCK:
        if _MARKETS and (time.time() - _MARKETS_TS) < _MARKETS_TTL:
            return _MARKETS
        try:
            markets = _client().load_markets()
            _MARKETS = set(markets)
            _MARKETS_TS = time.time()
        except Exception:
            pass  # keep the stale (possibly empty) set; callers degrade to skip
    return _MARKETS


def has_futures(symbol: str) -> bool:
    mk = _load_markets()
    return _unified(symbol) in mk if mk else False


def clear_cache() -> None:
    _CACHE.clear()


def fetch_derivatives(symbol: str) -> Optional[Dict[str, Any]]:
    """Positioning snapshot for one spot symbol, or None (no future / errors).

    {
      funding_rate: current 8h funding (fraction, e.g. 0.0001 = 0.01%),
      funding_hist: last ~30 funding rates (oldest→newest),
      oi_change_pct: fractional OI change over DERIV_OI_WINDOW_H hours,
      top_position_ratio: top-trader long/short POSITION ratio,
      global_account_ratio: global long/short ACCOUNT ratio,
    }
    """
    now = time.time()
    hit = _CACHE.get(symbol)
    if hit is not None and (now - hit[0]) < config.DERIV_TTL_SECONDS:
        return hit[1]
    if not has_futures(symbol):
        return None

    ex = _client()
    uni = _unified(symbol)
    raw = symbol.upper()
    try:
        with _FETCH_LOCK:  # ccxt sync client: one caller at a time
            fr = ex.fetch_funding_rate_history(uni, limit=30)
            funding_hist = [float(x.get("fundingRate") or 0.0) for x in fr] or [0.0]

            hours = max(2, config.DERIV_OI_WINDOW_H)
            oi = ex.fetch_open_interest_history(uni, timeframe="1h", limit=hours + 1)
            oi_vals = [float(x.get("openInterestAmount") or x.get("openInterestValue") or 0.0)
                       for x in oi]
            oi_vals = [v for v in oi_vals if v > 0]
            oi_change = ((oi_vals[-1] - oi_vals[0]) / oi_vals[0]) if len(oi_vals) >= 2 else 0.0

            top = ex.fapiDataGetTopLongShortPositionRatio(
                {"symbol": raw, "period": "1h", "limit": 1})
            top_ratio = float(top[-1]["longShortRatio"]) if top else 1.0
            acct = ex.fapiDataGetGlobalLongShortAccountRatio(
                {"symbol": raw, "period": "1h", "limit": 1})
            acct_ratio = float(acct[-1]["longShortRatio"]) if acct else 1.0
    except Exception:
        return None

    snap = {
        "funding_rate": funding_hist[-1],
        "funding_hist": funding_hist,
        "oi_change_pct": float(oi_change),
        "top_position_ratio": top_ratio,
        "global_account_ratio": acct_ratio,
    }
    _CACHE[symbol] = (now, snap)
    return snap


def funding_z(funding_hist: List[float]) -> float:
    """z-score of the latest funding vs its own recent history, tanh-squashed."""
    if not funding_hist:
        return 0.0
    cur = funding_hist[-1]
    hist = funding_hist[:-1] or [cur]
    mean = sum(hist) / len(hist)
    var = sum((x - mean) ** 2 for x in hist) / max(len(hist) - 1, 1)
    sd = math.sqrt(var)
    if sd <= 1e-12:
        return 0.0
    return math.tanh((cur - mean) / sd / 2.0)
