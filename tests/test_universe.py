"""v3.2: universe single-source-of-truth + env add/remove."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from universe import BASE_TICKERS, SYMBOLS, _merge


def test_universe_is_48_and_relic_free():
    assert len(SYMBOLS) == 48 and len(set(SYMBOLS)) == 48
    dead = {"MATICUSDT", "LUNAUSDT", "LRCUSDT", "MKRUSDT", "WAVESUSDT", "TONUSDT"}
    assert not dead & set(SYMBOLS)
    assert "POLUSDT" in SYMBOLS and "SUIUSDT" in SYMBOLS       # the renames/swaps


def test_consumers_derive_from_universe():
    from cycle import SYMBOLS as cycle_symbols
    from ingestion import KNOWN_TICKERS
    assert cycle_symbols is SYMBOLS                            # re-export, not a copy
    assert set(BASE_TICKERS) <= KNOWN_TICKERS                  # every pair news-taggable
    assert {"SPX", "DXY"} <= KNOWN_TICKERS                     # macro proxies preserved


def test_merge_add_remove_dedup_case():
    out = _merge(["BTC", "ETH"], " aptusdt, SEIUSDT ,APTUSDT", "ethusdt")
    assert out == ["BTCUSDT", "APTUSDT", "SEIUSDT"]            # dedup, case, remove
    assert _merge(["BTC"], "", "") == ["BTCUSDT"]
    assert _merge(["BTC"], "", "BTCUSDT") == []                # full self-removal allowed


def test_base_tickers_derived():
    assert BASE_TICKERS == [s[:-4] for s in SYMBOLS]
