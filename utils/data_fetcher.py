from __future__ import annotations
import os
import pandas as pd
from typing import Optional

class DataFetcher:
    """
    Loads OHLCV either from CSV files in ./data or live via ccxt.
    CSV format expected columns:
      timestamp (iso or epoch-ms), open, high, low, close, volume
    File name convention:
      data/{SYMBOL}_{TIMEFRAME}.csv  e.g., BTCUSDT_1h.csv
    """

    def __init__(self, prefer_csv: bool = True):
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
        import ccxt
        ex = getattr(ccxt, exchange_name)()

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

        # If prefer_csv = False → always fetch live
        return self.fetch_ccxt(symbol, timeframe, limit=limit)
