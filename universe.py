"""Trading universe — the single source of truth for which pairs the system
analyzes. Everything derives from here: the cycle iterates SYMBOLS, news
tagging derives KNOWN_TICKERS from BASE_TICKERS, preflight liveness-checks
every symbol at deploy.

Adding a coin is ONE line: either append the base ticker to _CORE below, or
set UNIVERSE_ADD=APTUSDT,SEIUSDT in .env (no code change, survives pulls).
Removing works the same via UNIVERSE_REMOVE. Everything downstream adapts
automatically — the RL policies are per-AGENT (global), not per-pair, so a new
pair is scored, graded and learned from starting with its first cycle;
history for backtests is fetched on demand. Optional per-coin polish:
ingestion.ALIASES (full-name headline matching) and research ECOSYSTEMS
membership. scripts/preflight.py fails loudly if an added symbol is not
actually trading (the LUNA/LRC/MKR lesson).

Audited 2026-07-04: all 48 pairs TRADING with fresh 1h candles on Binance
spot (MATIC→POL rename long since applied; TON still halted — do not add).
"""
from __future__ import annotations

import os
from typing import List

_CORE = [
    "AAVE", "ADA", "ALGO", "AR", "ARB", "ATOM", "AVAX", "AXS", "BCH", "BNB",
    "BTC", "CAKE", "COMP", "CRV", "DOGE", "DOT", "DYDX", "ENJ", "ETC", "ETH",
    "FET", "FIL", "FLOW", "GALA", "GMT", "GRT", "ICP", "IMX", "INJ", "LINK",
    "LTC", "MANA", "NEAR", "OP", "POL", "PYTH", "RENDER", "SAND",
    "SHIB", "SNX", "SOL", "STORJ", "SUI", "THETA", "TRX", "UNI", "WLD", "XRP",
]


def _merge(core: List[str], add_csv: str, remove_csv: str) -> List[str]:
    """Core tickers + UNIVERSE_ADD symbols − UNIVERSE_REMOVE symbols.
    Pure function (unit-tested directly). Add/remove entries are FULL symbols
    (e.g. APTUSDT); order is stable: core order, then adds append."""
    add = [s.strip().upper() for s in add_csv.split(",") if s.strip()]
    remove = {s.strip().upper() for s in remove_csv.split(",") if s.strip()}
    out, seen = [], set()
    for sym in [t + "USDT" for t in core] + add:
        if sym in remove or sym in seen:
            continue
        seen.add(sym)
        out.append(sym)
    return out


SYMBOLS: List[str] = _merge(_CORE, os.getenv("UNIVERSE_ADD", ""),
                            os.getenv("UNIVERSE_REMOVE", ""))

# Base assets (news tagging etc.) — derived, never hand-maintained.
BASE_TICKERS: List[str] = [s[:-4] for s in SYMBOLS if s.endswith("USDT")]
