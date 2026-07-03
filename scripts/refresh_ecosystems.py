#!/usr/bin/env python
"""Refresh ecosystem membership from CoinGecko categories (enhancement B4).

Hardcoded ecosystem dicts rot (the LUNA class of failure). This script maps
~15 CoinGecko categories onto our ecosystem names, intersects members with the
trading universe, and writes logs/ecosystems_cache.json. The research agent
uses the cache when ECOSYSTEMS_AUTO is on AND the cache is <7 days old —
otherwise the hardcoded dict. ECOSYSTEM_DRIVERS stay hardcoded (they encode
priority judgment, not membership).

Free tier budget: ~15 calls per run, run nightly => ~450/month vs 10k allowed.

    python scripts/refresh_ecosystems.py
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests

# ecosystem name (ours) -> CoinGecko category id
CATEGORY_MAP = {
    "ethereum": "ethereum-ecosystem",
    "solana": "solana-ecosystem",
    "bnb": "binance-smart-chain",
    "ai": "artificial-intelligence",
    "gaming": "gaming",
    "defi": "decentralized-finance-defi",
    "layer2": "layer-2",
    "meme": "meme-token",
    "storage": "storage",
    "oracle": "oracle",
}


def refresh(cache_path: str | None = None, session: requests.Session | None = None) -> dict:
    import config
    from cycle import SYMBOLS

    cache_path = cache_path or config.ECOSYSTEMS_CACHE_PATH
    sess = session or requests.Session()
    universe = {s[:-4] for s in SYMBOLS}          # base tickers
    out: dict = {}

    for eco, category in CATEGORY_MAP.items():
        try:
            r = sess.get(
                "https://api.coingecko.com/api/v3/coins/markets",
                params={"vs_currency": "usd", "category": category,
                        "order": "market_cap_desc", "per_page": 50, "page": 1},
                timeout=10)
            r.raise_for_status()
            symbols = [str(c.get("symbol", "")).upper() for c in r.json()]
            members = [s for s in symbols if s in universe]
            if members:
                out[eco] = members
            time.sleep(1.5)   # free-tier politeness
        except Exception as e:
            print(f"  {eco}: FAILED ({e}) — keeping previous/hardcoded")

    payload = {"fetched_ts": time.time(), "ecosystems": out}
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[ecosystems] wrote {len(out)} ecosystems -> {cache_path}")
    return payload


if __name__ == "__main__":
    refresh()
