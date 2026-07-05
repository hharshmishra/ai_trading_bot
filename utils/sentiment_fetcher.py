"""Free market-sentiment + on-chain + order-flow data (v3.5). $0, keyless.

Two feeds, cached separately:

* ``fetch_market_sentiment()`` — ONE market-wide bundle per SENTIMENT_TTL
  (55 min, so the 4h/1d cascade reuses the hourly fetch): Fear & Greed level
  + history (alternative.me), BTC mempool-size / n-transactions /
  estimated-tx-volume 30d series (blockchain.info charts), CoinGecko trending
  tickers. Each source degrades independently to None — a dead API costs one
  bundle field, never the cycle.

* ``fetch_taker_flow(symbol, timeframe)`` — per-pair raw Binance klines
  (public /api/v3/klines, 12-field rows; field 9 = taker-buy base volume,
  which ccxt's unified fetch_ohlcv discards). Closed candles only, matching
  CLOSED_CANDLES_ONLY. Side-channel fetch: the 6-col OHLCV contract in
  data_fetcher/backtest is never touched (derivatives_fetcher precedent).
"""
from __future__ import annotations

import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import requests

import config

FNG_URL = "https://api.alternative.me/fng/"
CHART_URL = "https://api.blockchain.info/charts/{name}"
TRENDING_URL = "https://api.coingecko.com/api/v3/search/trending"
KLINES_URL = "https://api.binance.com/api/v3/klines"

_TIMEOUT = 8.0
_TAKER_TTL = 300.0            # per (symbol, tf), like the live OHLCV cache
_NEG_TTL = 300.0              # failed market bundle: don't re-hit every decide

_LOCK = threading.Lock()
_MARKET_CACHE: Dict[str, tuple] = {}                 # "market" -> (ts, bundle|None)
_TAKER_CACHE: Dict[Tuple[str, str], tuple] = {}      # (sym, tf) -> (ts, rows)


def clear_cache() -> None:
    _MARKET_CACHE.clear()
    _TAKER_CACHE.clear()


def _get_json(url: str, params: Optional[dict] = None) -> Any:
    r = requests.get(url, params=params or {}, timeout=_TIMEOUT,
                     headers={"User-Agent": "BitReinforceX/1.0"})
    r.raise_for_status()
    return r.json()


def _fng() -> Tuple[Optional[float], Optional[List[float]]]:
    """(current 0..100, daily history oldest->newest, ~45d)."""
    data = _get_json(FNG_URL, {"limit": 45, "format": "json"})["data"]
    vals = [float(d["value"]) for d in data]          # API returns newest-first
    vals.reverse()
    return (vals[-1] if vals else None), (vals or None)


def _chart(name: str, timespan: str = "30days") -> Optional[List[float]]:
    """blockchain.info chart y-series (chronological)."""
    js = _get_json(CHART_URL.format(name=name),
                   {"timespan": timespan, "format": "json"})
    vals = [float(p["y"]) for p in js.get("values") or []]
    return vals or None


def _trending() -> Optional[set]:
    js = _get_json(TRENDING_URL)
    coins = js.get("coins") or []
    out = {str((c.get("item") or {}).get("symbol") or "").upper() for c in coins}
    out.discard("")
    return out or None


def fetch_market_sentiment() -> Optional[Dict[str, Any]]:
    """Market-wide bundle or None (only when EVERY source failed).

    { fng: float|None, fng_hist: [float]|None (oldest->newest),
      mempool: [float]|None, ntx: [float]|None, txvol: [float]|None,
      trending: set[str]|None }
    """
    now = time.time()
    hit = _MARKET_CACHE.get("market")
    if hit is not None:
        ts, bundle = hit
        ttl = config.SENTIMENT_TTL_SECONDS if bundle is not None else _NEG_TTL
        if (now - ts) < ttl:
            return bundle

    with _LOCK:
        hit = _MARKET_CACHE.get("market")
        if hit is not None and (time.time() - hit[0]) < _NEG_TTL:
            return hit[1]
        bundle: Dict[str, Any] = {"fng": None, "fng_hist": None, "mempool": None,
                                  "ntx": None, "txvol": None, "trending": None}
        try:
            bundle["fng"], bundle["fng_hist"] = _fng()
        except Exception:
            pass
        for key, chart in (("mempool", "mempool-size"),
                           ("ntx", "n-transactions"),
                           ("txvol", "estimated-transaction-volume-usd")):
            try:
                bundle[key] = _chart(chart)
            except Exception:
                pass
        try:
            bundle["trending"] = _trending()
        except Exception:
            pass

        if all(v is None for v in bundle.values()):
            bundle = None                              # total outage -> agent no-op
        _MARKET_CACHE["market"] = (time.time(), bundle)
        return bundle


def fetch_taker_flow(symbol: str, timeframe: str,
                     limit: int = 40) -> Optional[List[Tuple[float, float, float]]]:
    """Last ``limit`` CLOSED candles as (open_time_ms, volume, taker_buy_base).

    None on any failure — the agent's per-pair features degrade to 0.0.
    """
    key = (symbol.upper(), timeframe)
    now = time.time()
    hit = _TAKER_CACHE.get(key)
    if hit is not None and (now - hit[0]) < _TAKER_TTL:
        return hit[1]
    try:
        raw = _get_json(KLINES_URL, {"symbol": symbol.upper(),
                                     "interval": timeframe, "limit": limit})
        rows = [(float(k[0]), float(k[5]), float(k[9])) for k in raw]
        # field 6 = close time (ms): the last row is the in-progress candle
        # whenever its close time is still in the future — drop it (parity
        # with CLOSED_CANDLES_ONLY on the OHLCV path).
        if raw and float(raw[-1][6]) > now * 1000.0:
            rows = rows[:-1]
        if not rows:
            return None
    except Exception:
        return None
    _TAKER_CACHE[key] = (now, rows)
    return rows
