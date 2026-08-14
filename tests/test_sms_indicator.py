"""Smart Money Structure port (v3.8) — parity with the Pine mechanics.

Synthetic-OHLCV tests pin the semantics that matter: pivot confirmation lag,
colored-candle structure crossings (BOS/CHoCH), the vol-adaptive momentum +
volume + breakout BUY/SELL label, min-signal-distance, and the full-series ==
per-window property the backtest vectorization relies on.
"""
import numpy as np
import pandas as pd
import pytest

from agents import custom_indicators as ci


def _df(closes, highs=None, lows=None, opens=None, vols=None, freq="1h"):
    n = len(closes)
    closes = pd.Series([float(c) for c in closes])
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq=freq),
        "open": [float(o) for o in opens] if opens else closes.shift(1).fillna(closes[0]),
        "high": [float(h) for h in highs] if highs else closes + 0.1,
        "low": [float(l) for l in lows] if lows else closes - 0.1,
        "close": closes,
        "volume": [float(v) for v in vols] if vols else [100.0] * n,
    })


def _flat(n, price=100.0, vol=100.0):
    return {"closes": [price] * n, "highs": [price + 0.2] * n,
            "lows": [price - 0.2] * n, "opens": [price] * n, "vols": [vol] * n}


class TestPivots:
    def test_pivot_confirms_len_bars_late(self):
        b = _flat(80)
        b["highs"][10] = 110.0                       # lone pivot high at i=10
        d = ci.sms_structure(_df(**b), pivot_len=5)
        assert pd.isna(d["sms_last_high"].iloc[14])  # not yet confirmed
        assert d["sms_last_high"].iloc[15] == 110.0  # visible at i+len
        assert d["sms_last_high"].iloc[60] == 110.0  # ffilled state

    def test_too_short_returns_none(self):
        b = _flat(20)
        assert ci.sms_structure(_df(**b)) is None
        assert ci.sms_signal_from(_df(**b)) is None


class TestStructureEvents:
    def test_choch_sell_needs_cross_and_red_candle(self):
        b = _flat(80)
        b["highs"][10] = 110.0                        # last_high = 110 from i=15
        for i in range(16, 30):                       # price riding above 110
            b["lows"][i] = 111.0
            b["closes"][i] = 112.0
            b["opens"][i] = 112.0
            b["highs"][i] = 112.5
        b["lows"][30] = 108.0                         # cross under 110
        b["opens"][30] = 112.0
        b["closes"][30] = 109.0                       # red
        b["highs"][30] = 112.0
        d = ci.sms_structure(_df(**b), pivot_len=5)
        assert bool(d["sms_choch_sell"].iloc[30])
        # same cross with a green candle must NOT fire
        b["closes"][30] = 113.0
        b["highs"][30] = 113.5
        d2 = ci.sms_structure(_df(**b), pivot_len=5)
        assert not bool(d2["sms_choch_sell"].iloc[30])

    def test_bos_buy_breaks_previous_pivot_high(self):
        b = _flat(80)
        b["highs"][10] = 110.0
        b["lows"][30] = 108.0                         # keep other bars below 110
        b["highs"][40] = 111.0                        # cross over prev last_high
        b["opens"][40] = 100.0
        b["closes"][40] = 110.5                       # green
        d = ci.sms_structure(_df(**b), pivot_len=5)
        assert bool(d["sms_bos_buy"].iloc[40])


class TestBuySellLabel:
    def _burst(self, n=120, at=None):
        b = _flat(n)
        at = n - 1 if at is None else at
        b["opens"][at] = 100.0
        b["closes"][at] = 103.0                       # +3% momentum burst
        b["highs"][at] = 103.5
        b["vols"][at] = 1000.0                        # volume expansion
        return b

    def test_label_fires_with_all_filters(self):
        b = self._burst()
        sig = ci.sms_signal_from(_df(**b))
        assert sig == {"signal": "buy", "confidence": 0.62, "name": "sms"}

    def test_no_volume_no_label(self):
        b = self._burst()
        b["vols"][-1] = 100.0                         # no expansion
        d = ci.sms_structure(_df(**b))
        assert not bool(d["sms_buy"].iloc[-1])

    def test_no_breakout_no_label(self):
        b = self._burst()
        b["highs"][115] = 120.0     # inside the breakout_len=5 prior window
        d = ci.sms_structure(_df(**b))
        assert not bool(d["sms_buy"].iloc[-1])

    def test_min_distance_suppresses_rapid_refire(self):
        b = self._burst(n=120, at=100)
        # second full-confluence burst 3 bars later (< min_dist 5)
        b["opens"][103] = 103.0
        b["closes"][103] = 106.5
        b["highs"][103] = 107.0
        b["vols"][103] = 1000.0
        for i in (101, 102):
            b["closes"][i] = 103.0
            b["opens"][i] = 103.0
            b["highs"][i] = 103.2
            b["lows"][i] = 102.8
        d = ci.sms_structure(_df(**b), min_dist=5)
        assert bool(d["sms_buy"].iloc[100])
        assert not bool(d["sms_buy"].iloc[103])
        d2 = ci.sms_structure(_df(**b), min_dist=2)   # relaxed distance refires
        assert bool(d2["sms_buy"].iloc[103])


class TestWindowParity:
    def test_full_series_equals_trailing_window(self):
        """Same property the NWE vectorization proved: every computation is
        causal with bounded lookback, so the event on bar t computed over the
        full series matches sms_signal_from on a 150-bar trailing window."""
        rng = np.random.default_rng(42)
        n = 300
        closes = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
        spread = np.abs(rng.normal(0, 0.004, n))
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="1h"),
            "open": np.roll(closes, 1), "close": closes,
            "high": closes * (1 + spread), "low": closes * (1 - spread),
            "volume": rng.uniform(50, 500, n)})
        df.loc[0, "open"] = closes[0]

        full = ci.sms_structure(df)
        for t in range(160, n, 7):
            w = df.iloc[t - 149: t + 1].reset_index(drop=True)
            sig = ci.sms_signal_from(w)
            row = full.iloc[t]
            expected = None
            if row["sms_buy"]:
                expected = ("buy", "sms")
            elif row["sms_sell"]:
                expected = ("sell", "sms")
            elif row["sms_bos_buy"]:
                expected = ("buy", "sms_bos")
            elif row["sms_bos_sell"]:
                expected = ("sell", "sms_bos")
            elif row["sms_choch_buy"]:
                expected = ("buy", "sms_choch")
            elif row["sms_choch_sell"]:
                expected = ("sell", "sms_choch")
            got = (sig["signal"], sig["name"]) if sig else None
            assert got == expected, f"bar {t}: window {got} vs full {expected}"


class TestTrendMatrix:
    class _Fetcher:
        def __init__(self, direction=1):
            self.direction = direction

        def get_ohlcv(self, symbol, tf, limit=200):
            n = max(int(limit), 120)
            step = 0.5 * self.direction
            closes = pd.Series(100.0 + step * np.arange(n))
            return pd.DataFrame({
                "timestamp": pd.date_range("2024-01-01", periods=n, freq="1h"),
                "open": closes - 0.1 * self.direction, "close": closes,
                "high": closes + 0.2, "low": closes - 0.2,
                "volume": [100.0] * n})

    def test_aligned_uptrend(self):
        m = ci.sms_trend_matrix(self._Fetcher(1), "BTCUSDT")
        assert m["trend"] == {"1h": 1, "4h": 1, "1d": 1}
        assert m["strength"] == 100.0 and m["confidence"] == 90.0
        assert m["cvd_norm"] == pytest.approx(1.0)

    def test_aligned_downtrend(self):
        m = ci.sms_trend_matrix(self._Fetcher(-1), "BTCUSDT")
        assert m["strength"] == -100.0 and m["confidence"] == 90.0
        assert m["cvd_norm"] == pytest.approx(-1.0)

    def test_fetch_failure_returns_none(self):
        class Boom:
            def get_ohlcv(self, *a, **k):
                raise RuntimeError("net down")
        assert ci.sms_trend_matrix(Boom(), "BTCUSDT") is None


class TestAgentWiring:
    def _burst_df(self):
        """Realistic random walk (the full raw-indicator stack needs variance)
        with a full-confluence SMS buy burst engineered onto the last bar."""
        rng = np.random.default_rng(7)
        n = 620
        closes = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.008, n)))
        spread = np.abs(rng.normal(0, 0.003, n))
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="1h"),
            "open": np.roll(closes, 1), "close": closes,
            "high": closes * (1 + spread), "low": closes * (1 - spread),
            "volume": rng.uniform(50, 500, n)})
        df.loc[0, "open"] = closes[0]
        prior_hi = float(df["high"].iloc[-7:-1].max())
        close = prior_hi * 1.03                      # momentum + breakout
        df.loc[n - 1, "open"] = float(df["close"].iloc[-2])
        df.loc[n - 1, "close"] = close
        df.loc[n - 1, "high"] = close * 1.001
        df.loc[n - 1, "low"] = float(df["close"].iloc[-2]) * 0.999
        df.loc[n - 1, "volume"] = 5000.0             # volume expansion
        return df

    def test_sms_in_direct_signals_when_enabled(self, monkeypatch):
        import config
        from agents.indicator_agent import IndicatorAgent
        monkeypatch.setattr(config, "SMS_ENABLED", True)
        agent = IndicatorAgent()
        dec = agent.decide("TESTUSDT", "1h", ohlcv=self._burst_df(), log=False)
        names = [str(d.get("name")) for d in dec.details["direct_signals"]]
        # which SMS variant wins depends on the walk (min-dist can hand the
        # bar to BOS/CHoCH); the wiring contract is that ONE sms source fired
        assert any(n.startswith("sms") for n in names)
        # ohlcv was passed (backtest shape) -> matrix must NOT be fetched
        assert dec.details["sms"] is None

    def test_sms_absent_when_disabled(self, monkeypatch):
        import config
        from agents.indicator_agent import IndicatorAgent
        monkeypatch.setattr(config, "SMS_ENABLED", False)
        agent = IndicatorAgent()
        dec = agent.decide("TESTUSDT", "1h", ohlcv=self._burst_df(), log=False)
        names = [d.get("name") for d in dec.details["direct_signals"]]
        assert not any(str(x).startswith("sms") for x in names)

    def test_pick_sms_signal_reads_all_sms_names(self):
        from signals import pick_sms_signal
        for name in ("sms", "sms_bos", "sms_choch"):
            blk = {"raw": {"details": {"direct_signals": [
                {"name": name, "signal": "sell"}]}}}
            assert pick_sms_signal(blk) == "sell"
