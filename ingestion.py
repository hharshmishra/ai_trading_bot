"""News ingestion: pull free headlines, normalize, tag assets, feed RAG.

Network fetchers are injectable so the pipeline is unit-testable offline. All
default sources are RSS via feedparser — CryptoPanic's free API tier was
discontinued in April 2026, so it is no longer part of the default path (the
fetcher remains for anyone with a paid token).
"""
from __future__ import annotations

import calendar
import hashlib
import logging
import time
from typing import Any, Callable, Dict, List, Optional

from rag import RagIndex

logger = logging.getLogger("ingestion")

DEFAULT_RSS = [
    ("coindesk", "https://www.coindesk.com/arc/outboundfeeds/rss/"),
    ("cointelegraph", "https://cointelegraph.com/rss"),
    ("theblock", "https://www.theblock.co/rss.xml"),
    ("decrypt", "https://decrypt.co/feed"),
    ("bitcoinmagazine", "https://bitcoinmagazine.com/feed"),
    ("cryptoslate", "https://cryptoslate.com/feed/"),
    ("blockworks", "https://blockworks.co/feed"),
    ("newsbtc", "https://www.newsbtc.com/feed/"),
]

# Base tickers we track (trading universe + macro proxies).
# Derived from the trading universe (universe.py) so a newly added pair is
# news-taggable automatically; SPX/DXY are macro proxies the research agent
# tracks via tagged headlines. ALIASES below stay manual — full-name matching
# is editorial judgment, not derivable.
from universe import BASE_TICKERS

KNOWN_TICKERS = set(BASE_TICKERS) | {"SPX", "DXY"}

# Common name -> ticker, for tagging headlines that use full names.
ALIASES = {
    "bitcoin": "BTC", "ethereum": "ETH", "solana": "SOL", "cardano": "ADA",
    "ripple": "XRP", "dogecoin": "DOGE", "avalanche": "AVAX", "chainlink": "LINK",
    "polkadot": "DOT", "polygon": "POL", "shiba": "SHIB", "render": "RENDER",
    "arbitrum": "ARB", "optimism": "OP", "cosmos": "ATOM", "aave": "AAVE",
    "uniswap": "UNI", "litecoin": "LTC", "near protocol": "NEAR",
    "sui network": "SUI",   # bare "sui" would substring-match "lawsuit" etc.
}


# Source credibility tiers (enhancement C2) — 1 = most credible. Unknown
# sources default to tier 2.
SOURCE_TIERS = {
    "coindesk": 1, "theblock": 1, "blockworks": 1,
    "cointelegraph": 2, "decrypt": 2, "cryptoslate": 2,
    "newsbtc": 3, "bitcoinmagazine": 3,
}


def source_tier(source: str) -> int:
    return SOURCE_TIERS.get((source or "").lower(), 2)


def format_headline(row: Dict[str, Any], now: Optional[float] = None) -> str:
    """'[3h ago] [tier-1] Title' — the age/tier prefix lets the LLM weight
    recency and credibility (C2)."""
    now = now if now is not None else time.time()
    title = row.get("title") or ""
    pub = row.get("published_ts")
    if pub:
        age_h = max(0, int((now - float(pub)) // 3600))
        age = f"{age_h}h ago" if age_h < 48 else f"{age_h // 24}d ago"
    else:
        age = "undated"
    return f"[{age}] [tier-{source_tier(row.get('source'))}] {title}"


def _id_for(url: str, title: str) -> str:
    return hashlib.sha256(((url or "") + "|" + (title or "")).encode()).hexdigest()[:32]


def tag_assets(text: str) -> List[str]:
    """Tag a headline with base tickers. Short tickers (<=3 chars, e.g. OP/AR)
    only match in $TICKER or TICKERUSDT forms to avoid English-word false hits;
    longer tickers match on word boundaries; names match via ALIASES."""
    up = (text or "").upper()
    low = (text or "").lower()
    padded = f" {up} "
    hits = set()
    for t in KNOWN_TICKERS:
        if f"${t}" in up or f"{t}USDT" in up or f"{t}/USD" in up:
            hits.add(t)
        elif len(t) >= 4 and f" {t} " in padded:
            hits.add(t)
    for name, t in ALIASES.items():
        if name in low:
            hits.add(t)
    return sorted(hits)


def normalize_rss(entries: List[Dict[str, Any]], source: str) -> List[Dict[str, Any]]:
    items = []
    for e in entries:
        title = e.get("title") or ""
        body = e.get("summary") or e.get("description") or ""
        url = e.get("link") or ""
        pub = e.get("published_parsed")
        # feedparser normalizes published_parsed to UTC; timegm reads it as UTC
        # (mktime would read it as LOCAL — on an IST box every headline landed
        # 5.5h early, skewing ages and the 48h/7d windows).
        published_ts = calendar.timegm(pub) if pub else None
        items.append({
            "id": _id_for(url, title), "source": source, "title": title, "body": body,
            "url": url, "published_ts": published_ts, "assets": tag_assets(title + " " + body)})
    return items


def fetch_rss(url: str, timeout: float = 10.0) -> List[Dict[str, Any]]:
    """Fetch the feed body with a bounded timeout, then hand BYTES to
    feedparser. feedparser.parse(url) does its own unbounded urllib fetch — one
    dead feed host could pin an ingest thread forever and, over enough cycles,
    exhaust the to_thread pool and stall the scheduler."""
    import feedparser
    import requests
    try:
        r = requests.get(url, timeout=timeout,
                         headers={"User-Agent": "BitReinforceX/1.0 (+rss ingest)"})
        r.raise_for_status()
        return [dict(e) for e in feedparser.parse(r.content).entries]
    except Exception as e:
        logger.warning("rss fetch failed for %s: %s", url, e)
        return []


def fetch_cryptopanic(token: str, kind: str = "news") -> List[Dict[str, Any]]:
    import datetime
    import requests
    r = requests.get("https://cryptopanic.com/api/v1/posts/",
                     params={"auth_token": token, "public": "true", "kind": kind}, timeout=10)
    out = []
    for p in r.json().get("results", []):
        title, url, pub = p.get("title", ""), p.get("url", ""), p.get("published_at")
        ts = None
        if pub:
            try:
                ts = datetime.datetime.fromisoformat(pub.replace("Z", "+00:00")).timestamp()
            except Exception:
                ts = None
        currencies = [c.get("code") for c in (p.get("currencies") or []) if c.get("code")]
        assets = sorted({c.upper() for c in currencies} | set(tag_assets(title)))
        out.append({"id": _id_for(url, title), "source": "cryptopanic", "title": title,
                    "body": "", "url": url, "published_ts": ts, "assets": assets})
    return out


def ingest_all(index: RagIndex, rss_sources=DEFAULT_RSS, cryptopanic_token: Optional[str] = None,
               rss_fetcher: Callable[[str], List[Dict]] = fetch_rss,
               cp_fetcher: Callable[[str], List[Dict]] = fetch_cryptopanic,
               dedup_window_ts: Optional[float] = None) -> Dict[str, int]:
    items: List[Dict[str, Any]] = []
    for source, url in rss_sources:
        try:
            items.extend(normalize_rss(rss_fetcher(url), source))
        except Exception:
            pass
    if cryptopanic_token:
        try:
            items.extend(cp_fetcher(cryptopanic_token))
        except Exception:
            pass
    return index.ingest(items, dedup_window_ts=dedup_window_ts)
