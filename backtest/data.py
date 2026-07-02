"""Historical OHLCV download + CSV cache for the backtest harness.

Uses ccxt since-pagination (1000 candles/call, rate-limit sleeps) and caches to
``data/history/{SYMBOL}_{tf}.csv`` following the repo's existing data/ CSV
convention. Re-runs top up incrementally from the last cached candle.
"""
from __future__ import annotations

import os
import time
from typing import Optional

import pandas as pd

from utils.data_fetcher import _get_exchange

TF_MS = {"1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000, "1w": 604_800_000}
COLS = ["timestamp", "open", "high", "low", "close", "volume"]


def _to_ms(when: str | int | float) -> int:
    if isinstance(when, (int, float)):
        return int(when)
    return int(pd.Timestamp(when, tz="UTC").timestamp() * 1000)


def fetch_history(symbol: str, tf: str, since_ms: int, until_ms: Optional[int] = None,
                  exchange_name: str = "binance", page_limit: int = 1000) -> pd.DataFrame:
    """Paginated OHLCV download [since_ms, until_ms). Timestamps kept as epoch-ms."""
    ex = _get_exchange(exchange_name)
    tf_ms = TF_MS[tf]
    until_ms = until_ms or int(time.time() * 1000)
    rows = []
    cursor = int(since_ms)
    while cursor < until_ms:
        batch = ex.fetch_ohlcv(symbol, timeframe=tf, since=cursor, limit=page_limit)
        if not batch:
            break
        rows.extend(b for b in batch if b[0] < until_ms)
        last_ts = batch[-1][0]
        nxt = last_ts + tf_ms
        if nxt <= cursor:  # defensive: no forward progress
            break
        cursor = nxt
        if len(batch) < page_limit:
            break
        time.sleep(max(getattr(ex, "rateLimit", 200), 100) / 1000.0)

    df = pd.DataFrame(rows, columns=COLS).drop_duplicates(subset=["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)


def load_or_fetch(symbol: str, tf: str, start: str, end: Optional[str] = None,
                  cache_dir: str = "data/history") -> pd.DataFrame:
    """Cached history as a chronological DataFrame with datetime timestamps.

    The final (possibly still-open) candle is dropped so the replay only ever
    sees closed bars — same guarantee the live scheduler has at :30.
    """
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{symbol}_{tf}.csv")
    start_ms, end_ms = _to_ms(start), _to_ms(end) if end else None

    cached = None
    if os.path.exists(path):
        cached = pd.read_csv(path)
        if not cached.empty:
            first = int(cached["timestamp"].iloc[0])
            last = int(cached["timestamp"].iloc[-1])
            changed = False
            # backfill: an earlier --start than the cache has must fetch the
            # missing head, not silently truncate the range
            if start_ms < first - TF_MS[tf]:
                head = fetch_history(symbol, tf, start_ms, first)
                if not head.empty:
                    cached = pd.concat([head, cached])
                    changed = True
            top_up_from = last + TF_MS[tf]
            if (end_ms or int(time.time() * 1000)) > top_up_from:
                fresh = fetch_history(symbol, tf, top_up_from, end_ms)
                if not fresh.empty:
                    cached = pd.concat([cached, fresh])
                    changed = True
            if changed:
                cached = (cached.drop_duplicates(subset=["timestamp"])
                          .sort_values("timestamp").reset_index(drop=True))
                cached.to_csv(path, index=False)
        if cached.empty:
            cached = None

    if cached is None:
        cached = fetch_history(symbol, tf, start_ms, end_ms)
        if cached.empty:
            raise RuntimeError(f"no history returned for {symbol} {tf}")
        cached.to_csv(path, index=False)

    df = cached[(cached["timestamp"] >= start_ms)]
    if end_ms:
        df = df[df["timestamp"] < end_ms]
    df = df.copy().reset_index(drop=True)
    if len(df) > 1:
        df = df.iloc[:-1].reset_index(drop=True)  # drop the (possibly open) last candle
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df
