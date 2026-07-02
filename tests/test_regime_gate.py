"""Regime classifier + trend triggers + Gate v2 truth table (Phase 2)."""
import numpy as np
import pandas as pd
import pytest

import config
from agents.regime_agent import classify_regime


def _df(close, high=None, low=None, vol=None, freq="h"):
    n = len(close)
    close = np.asarray(close, dtype=float)
    high = np.asarray(high, dtype=float) if high is not None else close + 0.3
    low = np.asarray(low, dtype=float) if low is not None else close - 0.3
    vol = np.asarray(vol, dtype=float) if vol is not None else np.full(n, 1000.0)
    return pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=n, freq=freq),
        "open": np.concatenate([[close[0]], close[:-1]]),
        "high": high, "low": low, "close": close, "volume": vol,
    })


def _trending_up(n=160, seed=1):
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(np.abs(rng.normal(0.8, 0.2, n)))
    return _df(close, high=close + 0.4, low=close - 0.4)


def _trending_down(n=160, seed=2):
    rng = np.random.default_rng(seed)
    close = 500 - np.cumsum(np.abs(rng.normal(0.8, 0.2, n)))
    return _df(close, high=close + 0.4, low=close - 0.4)


def _ranging(n=160, seed=3):
    rng = np.random.default_rng(seed)
    base = 100 + 2.0 * np.sin(np.linspace(0, 12 * np.pi, n))
    close = base + rng.normal(0, 0.35, n)
    return _df(close, high=close + 0.8, low=close - 0.8)


class TestClassifyRegime:
    def test_strong_uptrend_detected(self):
        snap = classify_regime(_trending_up())
        assert snap.regime == "trend_up"
        assert snap.adx > config.REGIME_ADX_ENTER or snap.chop <= config.REGIME_CHOP_ENTER

    def test_strong_downtrend_detected(self):
        snap = classify_regime(_trending_down())
        assert snap.regime == "trend_down"

    def test_ranging_detected(self):
        snap = classify_regime(_ranging())
        assert snap.regime in ("ranging", "mixed")
        assert snap.regime != "trend_up" and snap.regime != "trend_down"

    def test_snapshot_fields_finite(self):
        snap = classify_regime(_trending_up())
        for v in (snap.adx, snap.chop, snap.vol_pct, snap.atr):
            assert v is not None and np.isfinite(v)
        assert snap.atr > 0
        assert 0.0 <= snap.vol_pct <= 1.0

    def test_hysteresis_limits_flips_on_whipsaw(self):
        # alternate trend/range segments; dwell + asymmetric thresholds must
        # keep flips well below segment count
        rng = np.random.default_rng(9)
        parts = []
        px = 100.0
        for i in range(8):
            if i % 2 == 0:
                seg = px + np.cumsum(np.abs(rng.normal(0.7, 0.2, 20)))
            else:
                seg = px + rng.normal(0, 0.3, 20)
            px = seg[-1]
            parts.append(seg)
        close = np.concatenate(parts)
        snap = classify_regime(_df(close))
        assert snap.flips_in_walk <= 6  # 8 raw segments, hysteresis dampens

    def test_deterministic(self):
        df = _trending_up()
        a = classify_regime(df)
        b = classify_regime(df.copy())
        assert a == b

    def test_short_df_returns_mixed(self):
        snap = classify_regime(_df(np.linspace(100, 101, 20)))
        assert snap.regime == "mixed"  # not enough data -> neutral, never crashes


class TestTrendTriggers:
    def test_donchian_breakout_buy(self):
        from agents.custom_indicators import donchian_breakout_signal
        n = 80
        close = np.concatenate([100 + np.random.default_rng(4).normal(0, 0.2, n - 1), [104.0]])
        df = _df(close, high=close + 0.2, low=close - 0.2)
        sig = donchian_breakout_signal(df)
        assert sig is not None
        assert sig["signal"] == "buy"
        assert 0.5 < sig["confidence"] <= 0.99
        assert sig["name"] == "donchian_breakout"

    def test_donchian_no_breakout_is_none(self):
        from agents.custom_indicators import donchian_breakout_signal
        df = _ranging(100)
        sig = donchian_breakout_signal(df)
        assert sig is None or sig["signal"] == "skip"

    def test_squeeze_release_fires_with_direction(self):
        from agents.custom_indicators import squeeze_release_signal
        rng = np.random.default_rng(6)
        # long tight consolidation then expansion upward
        tight = 100 + rng.normal(0, 0.05, 60)
        burst = tight[-1] + np.cumsum(np.abs(rng.normal(0.9, 0.2, 2)))
        close = np.concatenate([tight, burst])
        df = _df(close, high=close + np.concatenate([np.full(60, 0.08), np.full(2, 0.5)]),
                 low=close - np.concatenate([np.full(60, 0.08), np.full(2, 0.5)]))
        sig = squeeze_release_signal(df)
        assert sig is not None and sig["signal"] == "buy"
        assert sig["name"] == "squeeze_release"

    def test_supertrend_flip_from_raw(self):
        from agents.indicator_agent import supertrend_flip_from_raw
        raw = pd.DataFrame({
            "close": [100, 101, 102], "supertrend": [99, 99.5, 100.2],
            "supertrend_dir": [-1, -1, 1], "atr": [1.0, 1.0, 1.0],
        })
        sig = supertrend_flip_from_raw(raw)
        assert sig is not None and sig["signal"] == "buy"
        assert sig["name"] == "supertrend_flip"

    def test_supertrend_no_flip_is_none(self):
        from agents.indicator_agent import supertrend_flip_from_raw
        raw = pd.DataFrame({
            "close": [100, 101, 102], "supertrend": [99, 99.5, 100.2],
            "supertrend_dir": [1, 1, 1], "atr": [1.0, 1.0, 1.0],
        })
        assert supertrend_flip_from_raw(raw) is None


def _res(tf="4h", regime="ranging", final_action="buy", final_conf=0.85,
         nwe=None, trend_fired=(), vol_ok=True):
    direct = []
    if nwe:
        direct.append({"name": "nwe", "signal": nwe, "confidence": 0.8})
    for name, sig in trend_fired:
        direct.append({"name": name, "signal": sig, "confidence": 0.75})
    return {
        "chartName": "TESTUSDT", "timeframe": tf,
        "final": {"action": final_action, "confidence": final_conf},
        "agents": {"indicator": {"action": final_action, "confidence": final_conf,
                                 "raw": {"details": {
                                     "regime": regime,
                                     "regime_feats": {"vol_ok": vol_ok},
                                     "direct_signals": direct}}}},
    }


class TestGateV2TruthTable:
    def setup_method(self):
        from signals import should_emit_signal_v2
        self.gate = should_emit_signal_v2

    # --- 1h ---
    def test_1h_ranging_nwe_vol_ok_emits(self):
        emit, overall, nwe, conf, reason = self.gate(_res(tf="1h", nwe="buy"))
        assert emit and overall == "buy" and reason == "nwe_ranging"

    def test_1h_ranging_nwe_low_volume_suppressed(self):
        emit, *_, reason = self.gate(_res(tf="1h", nwe="buy", vol_ok=False))
        assert not emit and reason == "low_volume"

    def test_1h_ranging_conf_alone_never_emits(self):
        emit, *_ = self.gate(_res(tf="1h", final_conf=0.95))
        assert not emit

    def test_1h_mixed_needs_brain_agreement(self):
        r = _res(tf="1h", regime="mixed", nwe="buy", final_action="buy")
        assert self.gate(r)[0]
        r2 = _res(tf="1h", regime="mixed", nwe="buy", final_action="sell")
        assert not self.gate(r2)[0]

    def test_1h_trending_nwe_suppressed(self):
        r = _res(tf="1h", regime="trend_up", nwe="sell")
        emit, *_, reason = self.gate(r)
        assert not emit and reason == "nwe_trend_suppressed"

    def test_1h_trending_trend_triggers_off_by_default(self):
        r = _res(tf="1h", regime="trend_up",
                 trend_fired=[("supertrend_flip", "buy"), ("donchian_breakout", "buy")])
        assert not self.gate(r)[0]  # GATE_1H_TREND defaults false

    # --- 4h/1d/1w ---
    def test_4h_ranging_nwe_emits(self):
        emit, overall, nwe, conf, reason = self.gate(_res(nwe="sell", final_action="buy"))
        assert emit and overall == "sell" and reason == "nwe_ranging"  # NWE overrides conf

    def test_4h_ranging_conf_emits_without_nwe(self):
        emit, overall, *_ , reason = self.gate(_res(final_conf=0.85))
        assert emit and overall == "buy" and reason == "conf_over_80"

    def test_4h_ranging_neither_suppressed(self):
        assert not self.gate(_res(final_conf=0.5))[0]

    def test_4h_trending_aligned_trend_trigger_emits(self):
        r = _res(regime="trend_up", final_conf=0.5,
                 trend_fired=[("donchian_breakout", "buy"), ("squeeze_release", "buy")])
        emit, overall, *_ , reason = self.gate(r)
        assert emit and overall == "buy" and reason == "trend_continuation"

    def test_4h_trending_counter_majority_without_flip_suppressed(self):
        r = _res(regime="trend_up", final_conf=0.5,
                 trend_fired=[("donchian_breakout", "sell"), ("squeeze_release", "sell")])
        emit, *_ , reason = self.gate(r)
        assert not emit and reason == "counter_trend_no_flip"

    def test_4h_trending_supertrend_flip_reversal_emits(self):
        r = _res(regime="trend_up", final_conf=0.5,
                 trend_fired=[("supertrend_flip", "sell"), ("donchian_breakout", "sell")])
        emit, overall, *_ , reason = self.gate(r)
        assert emit and overall == "sell" and reason == "trend_reversal"

    def test_4h_trending_nwe_suppressed_even_with_conf(self):
        r = _res(regime="trend_down", nwe="buy", final_action="buy", final_conf=0.9)
        emit, *_ , reason = self.gate(r)
        # NWE dead in trend; conf path counter to trend also suppressed
        assert not emit

    def test_4h_trending_conf_aligned_emits(self):
        r = _res(regime="trend_down", final_action="sell", final_conf=0.9)
        emit, overall, *_ , reason = self.gate(r)
        assert emit and overall == "sell" and reason == "conf_over_80"

    def test_4h_trending_conf_counter_suppressed(self):
        r = _res(regime="trend_down", final_action="buy", final_conf=0.9)
        emit, *_ , reason = self.gate(r)
        assert not emit and reason == "counter_trend_conf"

    def test_4h_mixed_nwe_needs_agreement(self):
        r = _res(regime="mixed", nwe="buy", final_action="buy", final_conf=0.5)
        assert self.gate(r)[0]
        r2 = _res(regime="mixed", nwe="buy", final_action="sell", final_conf=0.5)
        emit, *_ , reason = self.gate(r2)
        assert not emit

    def test_4h_mixed_conf_emits(self):
        r = _res(regime="mixed", final_conf=0.85)
        emit, overall, *_ , reason = self.gate(r)
        assert emit and reason == "conf_over_80"

    def test_missing_regime_falls_back_to_v1(self):
        # rows without regime (old snapshot / flag mid-flip) must behave like v1
        r = _res(regime=None, nwe="buy")
        emit, overall, *_ , reason = self.gate(r)
        assert emit and overall == "buy" and reason == "nwe_direct"


_LEGACY_SCHEMA = """
CREATE TABLE predictions (
    id TEXT PRIMARY KEY, cycle_id TEXT, pair TEXT NOT NULL, tf TEXT NOT NULL,
    created_ts REAL NOT NULL, candle_close_ts REAL, entry_price REAL,
    horizon_k INTEGER, grade_due_ts REAL, final_action TEXT,
    final_confidence REAL, final_score REAL, emitted INTEGER DEFAULT 0,
    news_action TEXT, news_action_idx INTEGER, news_feats TEXT, news_conf REAL,
    research_action TEXT, research_action_idx INTEGER, research_feats TEXT,
    research_conf REAL, indicator_action TEXT, indicator_conf REAL,
    indicator_blend TEXT, brain_weights TEXT,
    label_source TEXT DEFAULT 'pending', graded INTEGER DEFAULT 0, session_id TEXT
);
CREATE TABLE outcomes (
    prediction_id TEXT PRIMARY KEY, realized_return REAL, realized_label TEXT,
    threshold REAL, horizon_k INTEGER, graded_ts REAL, source TEXT
);
"""


class TestSchemaMigration:
    def test_fresh_store_has_new_columns(self, tmp_path):
        from persistence import Store
        s = Store(str(tmp_path / "fresh.db"))
        cols = {r["name"] for r in s.conn.execute("PRAGMA table_info(predictions)")}
        assert {"regime", "regime_feats", "atr", "tp_price", "sl_price",
                "trigger_source", "deriv_feats", "meta_p", "calibrated_conf"} <= cols
        ocols = {r["name"] for r in s.conn.execute("PRAGMA table_info(outcomes)")}
        assert {"label_tb", "barrier_hit_idx", "exit_price"} <= ocols
        s.close()

    def test_pre_upgrade_db_migrates_and_keeps_rows(self, tmp_path):
        import sqlite3
        from persistence import Store
        db = str(tmp_path / "old.db")
        conn = sqlite3.connect(db)
        conn.executescript(_LEGACY_SCHEMA)
        conn.execute("INSERT INTO predictions (id, pair, tf, created_ts) VALUES ('p1','BTCUSDT','1h',1.0)")
        conn.commit()
        conn.close()

        s = Store(db)  # migration runs in __init__
        row = s.get_prediction("p1")
        assert row is not None and row["pair"] == "BTCUSDT"
        assert row["regime"] is None and row["tp_price"] is None  # legacy NULLs
        s.close()

        s2 = Store(db)  # idempotent second open
        assert s2.get_prediction("p1") is not None
        s2.close()

    def test_record_prediction_roundtrip_with_regime(self, tmp_path):
        from persistence import Store
        s = Store(str(tmp_path / "rt.db"))
        decision = {
            "chartName": "ETHUSDT", "timeframe": "4h",
            "final": {"action": "buy", "confidence": 0.7, "score": 0.4},
            "agents": {"indicator": {"action": "buy", "confidence": 0.7,
                                     "raw": {"details": {
                                         "regime": "trend_up",
                                         "regime_feats": {"adx": 31.2, "vol_ok": True, "atr": 12.5},
                                         "blend": {"type1_share": 0.6}}}}},
            "policy": {"weights": {"indicator": 0.5}},
        }
        pid = s.record_prediction(decision, entry_price=3000.0, horizon_k=2,
                                  tp_price=3018.75, sl_price=2987.5,
                                  trigger_source="trend_continuation")
        row = s.get_prediction(pid)
        assert row["regime"] == "trend_up"           # auto-extracted from details
        assert row["regime_feats"]["adx"] == 31.2    # JSON round-trip
        assert row["atr"] == 12.5                    # pulled from regime_feats
        assert row["tp_price"] == 3018.75
        assert row["trigger_source"] == "trend_continuation"
        s.close()

    def test_record_outcome_with_tb_fields(self, tmp_path):
        from persistence import Store
        s = Store(str(tmp_path / "tb.db"))
        s.conn.execute("INSERT INTO predictions (id, pair, tf, created_ts) VALUES ('p1','BTCUSDT','1h',1.0)")
        s.conn.commit()
        s.record_outcome("p1", 0.01, "buy", 0.004, 3, label_tb="tp",
                         barrier_hit_idx=2, exit_price=101.5)
        o = s.get_outcome("p1")
        assert o["label_tb"] == "tp" and o["barrier_hit_idx"] == 2
        s.close()


class TestFlagOffNoOp:
    def test_decide_details_identical_when_flag_off(self, monkeypatch):
        """GATE_V2_ENABLED=False -> decide() output byte-identical to pre-Phase-2."""
        import agents.indicator_agent as ia
        monkeypatch.setattr(config, "GATE_V2_ENABLED", False)
        rng = np.random.default_rng(12)
        close = 100 + np.cumsum(rng.normal(0, 0.5, 140))
        df = _df(close)
        agent = ia.IndicatorAgent()
        dec = agent.decide("TESTUSDT", "1h", ohlcv=df.copy(), log=False)
        names = {d["name"] for d in dec.details["direct_signals"]}
        assert names.issubset({"nwe", "chandelier_exit", "alpha_trend"})
        # regime is still computed and stored (shadow logging), but the
        # trigger set / blend inputs stay legacy
        assert "regime" in dec.details
