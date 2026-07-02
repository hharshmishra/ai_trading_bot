"""Free macro sentiment/structure feeds (accuracy upgrade Phase 4).

Both are context features for the shared MarketContext (and later the
meta-label model) — never hard signal triggers. Failures return None and must
never sink a cycle. Both endpoints are keyless free tiers:
  - alternative.me Fear & Greed index (0..100, contrarian at extremes)
  - CoinGecko global BTC dominance (%; regime hint: majors vs alts)
"""
from __future__ import annotations

import time
from typing import Optional

import requests

_TTL = 30 * 60
_cache = {"fng": (0.0, None), "dom": (0.0, None)}


def fetch_fear_greed(timeout: float = 5.0) -> Optional[float]:
    now = time.time()
    ts, val = _cache["fng"]
    if val is not None and (now - ts) < _TTL:
        return val
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=1", timeout=timeout)
        r.raise_for_status()
        val = float(r.json()["data"][0]["value"])
        _cache["fng"] = (now, val)
        return val
    except Exception:
        return None


def fetch_btc_dominance(timeout: float = 5.0) -> Optional[float]:
    now = time.time()
    ts, val = _cache["dom"]
    if val is not None and (now - ts) < _TTL:
        return val
    try:
        r = requests.get("https://api.coingecko.com/api/v3/global", timeout=timeout)
        r.raise_for_status()
        val = float(r.json()["data"]["market_cap_percentage"]["btc"])
        _cache["dom"] = (now, val)
        return val
    except Exception:
        return None
