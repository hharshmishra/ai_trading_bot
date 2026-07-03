"""Real SPX / DXY price trends (enhancement B1).

Research logics 2 (SPX) and 5 (DXY) previously scored from news sentiment
ALONE — no price data at all. This module supplies an actual trend score from
free daily series:

  1. FRED (official, free API key in .env FRED_API_KEY): SP500, DTWEXBGS
  2. stooq.com keyless CSV fallback: ^spx; DXY has no reliable stooq symbol
     (dx.f coverage is spotty) so DXY is FRED-or-nothing — callers fall back
     to news-only scoring when this returns None.

Every failure returns None; nothing here can sink a cycle. 12h TTL cache
(daily series — refetching more often is pointless).
"""
from __future__ import annotations

import io
import os
import time
from typing import Optional

import pandas as pd
import requests

_TTL = 12 * 3600
_cache: dict = {}   # key -> (fetched_ts, Optional[pd.Series])

SPX_TREND_SCALE = 0.03   # 5-day SPX move of 3% saturates the score
DXY_TREND_SCALE = 0.02   # DXY moves are smaller — 2% saturates


def _cached(key: str, fetch) -> Optional[pd.Series]:
    now = time.time()
    hit = _cache.get(key)
    if hit is not None and (now - hit[0]) < _TTL:
        return hit[1]
    try:
        series = fetch()
    except Exception:
        series = None
    _cache[key] = (now, series)
    return series


def _fred_series(series_id: str, api_key: str, timeout: float = 8.0) -> Optional[pd.Series]:
    r = requests.get(
        "https://api.stlouisfed.org/fred/series/observations",
        params={"series_id": series_id, "api_key": api_key, "file_type": "json",
                "sort_order": "desc", "limit": 30},
        timeout=timeout)
    r.raise_for_status()
    obs = r.json().get("observations", [])
    vals = [(o["date"], float(o["value"])) for o in obs if o.get("value") not in (None, ".")]
    if not vals:
        return None
    s = pd.Series({pd.Timestamp(d): v for d, v in vals}).sort_index()
    return s


def _stooq_series(symbol: str, timeout: float = 8.0) -> Optional[pd.Series]:
    r = requests.get(f"https://stooq.com/q/d/l/?s={symbol}&i=d", timeout=timeout)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    if "Close" not in df.columns or df.empty:
        return None
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    s = df.dropna(subset=["Date"]).set_index("Date")["Close"].astype(float).sort_index()
    return s.tail(30) if len(s) else None


def _trend_score(series: Optional[pd.Series], days: int = 5,
                 scale: float = 0.03) -> Optional[float]:
    """clip(pct change over the last ``days`` sessions / scale, -1, 1)."""
    if series is None or len(series) < days + 1:
        return None
    cur, prev = float(series.iloc[-1]), float(series.iloc[-(days + 1)])
    if prev == 0:
        return None
    return float(max(-1.0, min(1.0, (cur - prev) / prev / scale)))


def spx_score() -> Optional[float]:
    """5-day S&P 500 trend in [-1, 1], or None when no source is reachable."""
    key = os.getenv("FRED_API_KEY")
    series = _cached("spx_fred", lambda: _fred_series("SP500", key)) if key else None
    if series is None:
        series = _cached("spx_stooq", lambda: _stooq_series("^spx"))
    return _trend_score(series, days=5, scale=SPX_TREND_SCALE)


def dxy_score() -> Optional[float]:
    """5-day broad-dollar trend in [-1, 1]; FRED-only (see module docstring)."""
    key = os.getenv("FRED_API_KEY")
    if not key:
        return None
    series = _cached("dxy_fred", lambda: _fred_series("DTWEXBGS", key))
    return _trend_score(series, days=5, scale=DXY_TREND_SCALE)
