from __future__ import annotations
import os
import time
import pandas as pd
from typing import Any, Dict, Optional, Tuple

# ---------------------------------------------------------------------------
# Per-cycle OHLCV cache + shared CCXT clients (Phase 1).
#
# A single analysis cycle fetches the same pairs many times: the indicator
# agent, the research agent's child/parent trends, and the shared market-context
# builder (money-flow basket, BTC/BTCDOM dominance, ecosystem drivers) all hit
# common pairs like BTCUSDT/ETHUSDT. Without caching, BTCUSDT alone is refetched
# dozens of times per cycle. This module-level TTL cache (shared across every
# DataFetcher instance) fetches each (symbol, timeframe, limit) once per cycle.
#
# We also reuse one ccxt exchange object per exchange instead of constructing a
# fresh client on every fetch (the old behaviour) — cutting setup overhead and
# easing Binance rate limits.
# ---------------------------------------------------------------------------
_OHLCV_TTL = float(os.getenv("OHLCV_TTL", "300"))  # seconds; one cycle << TTL
_OHLCV_CACHE: Dict[Tuple[str, str, int], Tuple[float, pd.DataFrame]] = {}
_EXCHANGES: Dict[str, Any] = {}


def clear_ohlcv_cache() -> None:
    """Drop all cached OHLCV. Call at the start of each cycle for freshness."""
    _OHLCV_CACHE.clear()


def _get_exchange(exchange_name: str = "binance"):
    ex = _EXCHANGES.get(exchange_name)
    if ex is None:
        import ccxt
        ex = getattr(ccxt, exchange_name)()
        _EXCHANGES[exchange_name] = ex
    return ex


class DataFetcher:
    """
    Loads OHLCV either from CSV files in ./data or live via ccxt.
    CSV format expected columns:
      timestamp (iso or epoch-ms), open, high, low, close, volume
    File name convention:
      data/{SYMBOL}_{TIMEFRAME}.csv  e.g., BTCUSDT_1h.csv
    """

    def __init__(self, prefer_csv: bool = False):
        self.prefer_csv = prefer_csv
        try:
            import ccxt  # noqa: F401
            self._ccxt_available = True
        except Exception:
            self._ccxt_available = False

        # ensure ./data folder exists
        if not os.path.exists("data"):
            os.makedirs("data")

    def _csv_path(self, symbol: str, timeframe: str) -> str:
        return f"data/{symbol}_{timeframe}.csv"

    def load_csv(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        path = self._csv_path(symbol, timeframe)
        if not os.path.exists(path):
            return None
        df = pd.read_csv(path)
        if "timestamp" in df.columns:
            # try to interpret epoch-ms or iso
            if pd.api.types.is_integer_dtype(df["timestamp"]):
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", errors="coerce")
            else:
                df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        else:
            raise ValueError("CSV must include a 'timestamp' column.")
        df = df.rename(columns=str.lower)
        df = df[["timestamp","open","high","low","close","volume"]]
        return df.dropna().sort_values("timestamp")

    def fetch_ccxt(self, symbol: str, timeframe: str, limit: int = 500, exchange_name: str = "binance") -> pd.DataFrame:
        if not self._ccxt_available:
            raise RuntimeError("ccxt not installed. Install from requirements.txt or set prefer_csv=True.")
        ex = _get_exchange(exchange_name)  # reuse one client per exchange

        # ccxt requires BTC/USDT style, not BTCUSDT
        ccxt_symbol = symbol

        raw = ex.fetch_ohlcv(ccxt_symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(raw, columns=["timestamp","open","high","low","close","volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df

    # def get_ohlcv(self, symbol: str, timeframe: str, limit: int = 500) -> pd.DataFrame:
    #     path = self._csv_path(symbol, timeframe)

    #     if self.prefer_csv:
    #         csv = self.load_csv(symbol, timeframe)
    #         if csv is not None:
    #             return csv.tail(limit)

    #         # if CSV missing → fetch live, save, then return
    #         df = self.fetch_ccxt(symbol, timeframe, limit=limit)
    #         df.to_csv(path, index=False)
    #         return df

    #     # If prefer_csv = False → always fetch live
    #     return self.fetch_ccxt(symbol, timeframe, limit=limit)

    def get_ohlcv(self, symbol: str, timeframe: str, limit: int = 500) -> pd.DataFrame:
        path = self._csv_path(symbol, timeframe)

        if self.prefer_csv:
            # Try loading existing CSV
            csv = self.load_csv(symbol, timeframe)
            if csv is None or csv.empty:
                df = self.fetch_ccxt(symbol, timeframe, limit=limit)
                df.to_csv(path, index=False)
                return df

            if csv is not None and not csv.empty:
                # Fetch latest candles
                new_df = self.fetch_ccxt(symbol, timeframe, limit=limit)

                # Merge & drop duplicates (by timestamp)
                df = pd.concat([csv, new_df]).drop_duplicates(subset=["timestamp"], keep="last")

                # Keep only last `limit` rows (rolling window)
                df = df.tail(limit)

                # Save back to CSV
                df.to_csv(path, index=False)
                return df

        # If prefer_csv = False → always fetch live, cached per (symbol, tf, limit)
        # within the TTL so repeated intra-cycle fetches of the same pair are free.
        # We hand out copies so callers that mutate the frame (e.g. add MA columns)
        # never corrupt the cached original.
        key = (symbol, timeframe, limit)
        now = time.time()
        hit = _OHLCV_CACHE.get(key)
        if hit is not None and (now - hit[0]) < _OHLCV_TTL:
            return hit[1].copy()
        df = self.fetch_ccxt(symbol, timeframe, limit=limit)
        _OHLCV_CACHE[key] = (now, df)
        return df.copy()
