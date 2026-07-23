"""Phase 5: nightly meta-label training + isotonic calibration. No network."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from jobs.features import FEATURE_NAMES, meta_features_from_prediction_row


def _insert_rows(store, n=1000, seed=7):
    """Synthetic graded predictions with a LEARNABLE pattern: trending-regime
    predictions with high agreement are usually correct; mixed-regime low-conf
    ones usually wrong."""
    rng = np.random.default_rng(seed)
    now = time.time()
    with store._lock:
        for i in range(n):
            good = i % 2 == 0
            regime = "trend_up" if good else "mixed"
            conf = 0.8 if good else 0.45
            correct = rng.random() < (0.75 if good else 0.3)
            action = "buy"
            realized = action if correct else "sell"
            tf = "1h" if i % 3 else "4h"
            pid = f"p{i}"
            store.conn.execute(
                "INSERT INTO predictions (id, pair, tf, created_ts, candle_close_ts, "
                "entry_price, final_action, final_confidence, graded, label_source, "
                "regime, regime_feats, atr, indicator_action, indicator_conf, emitted, "
                "trigger_source) VALUES (?,?,?,?,?,?,?,?,1,'auto',?,?,?,?,?,?,?)",
                (pid, "BTCUSDT", tf, now - (n - i) * 3600, now - (n - i) * 3600,
                 100.0, action, conf, regime,
                 json.dumps({"adx": 30.0 if good else 15.0, "chop": 35.0 if good else 60.0,
                             "vol_pct": 0.6, "atr": 1.2, "vol_ok": True}),
                 1.2, action, conf, 1, "nwe_ranging"))
            store.conn.execute(
                "INSERT INTO outcomes (prediction_id, realized_return, realized_label, "
                "threshold, horizon_k, graded_ts, source) VALUES (?,?,?,?,?,?,'auto')",
                (pid, 0.01 if correct else -0.01, realized, 0.004, 3, now - (n - i) * 3600 + 60))
        store.conn.commit()


@pytest.fixture
def artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "META_MODEL_PATH", str(tmp_path / "meta_model.pkl"))
    monkeypatch.setattr(config, "META_METRICS_PATH", str(tmp_path / "meta_metrics.json"))
    monkeypatch.setattr(config, "CALIBRATION_PATH", str(tmp_path / "calibration.json"))
    import jobs.nightly as jn
    jn._MODEL_CACHE.update({"path_mtime": None, "model": None})
    jn._CALIB_CACHE.update({"path_mtime": None, "knots": None})
    return tmp_path


class TestTraining:
    def test_full_training_run(self, tmp_path, artifacts):
        from persistence import Store
        from jobs.nightly import run_nightly_training
        store = Store(str(tmp_path / "t.db"))
        _insert_rows(store, n=1000)

        summary = run_nightly_training(store)
        assert summary["rows"] == 1000
        m = summary["meta"]
        assert m is not None and m["n_train"] == 800 and m["n_holdout"] == 200
        assert m["holdout_auc"] is not None and m["holdout_auc"] > 0.6  # learnable pattern
        assert Path(config.META_MODEL_PATH).exists()
        assert Path(config.META_METRICS_PATH).exists()
        assert "1h" in summary["calibrated_tfs"]
        store.close()

    def test_below_min_rows_is_noop(self, tmp_path, artifacts):
        from persistence import Store
        from jobs.nightly import train_meta_model
        store = Store(str(tmp_path / "s.db"))
        _insert_rows(store, n=100)
        assert train_meta_model(store.training_rows()) is None
        assert not Path(config.META_MODEL_PATH).exists()
        store.close()

    def test_meta_probability_inference(self, tmp_path, artifacts):
        from persistence import Store
        from jobs.nightly import meta_probability, run_nightly_training
        store = Store(str(tmp_path / "i.db"))
        _insert_rows(store, n=800)
        run_nightly_training(store)

        good_row = {"regime": "trend_up", "regime_feats": {"adx": 30, "chop": 35, "vol_pct": 0.6,
                                                           "atr": 1.2, "vol_ok": True},
                    "atr": 1.2, "entry_price": 100.0, "tf": "1h", "candle_close_ts": time.time(),
                    "final_action": "buy", "final_confidence": 0.8, "indicator_action": "buy",
                    "indicator_conf": 0.8, "emitted": 1, "trigger_source": "nwe_ranging"}
        bad_row = dict(good_row, regime="mixed", final_confidence=0.45, indicator_conf=0.45,
                       regime_feats={"adx": 15, "chop": 60, "vol_pct": 0.6, "atr": 1.2, "vol_ok": True})
        pg, pb = meta_probability(good_row), meta_probability(bad_row)
        assert pg is not None and pb is not None
        assert pg > pb  # model learned the planted pattern
        store.close()

    def test_meta_probability_none_without_artifact(self, artifacts):
        from jobs.nightly import meta_probability
        assert meta_probability({"tf": "1h"}) is None


class TestCalibration:
    def test_knots_monotonic_and_interp(self, tmp_path, artifacts):
        from persistence import Store
        from jobs.nightly import apply_calibration, fit_calibration
        store = Store(str(tmp_path / "c.db"))
        _insert_rows(store, n=600)
        payload = fit_calibration(store.training_rows())
        assert "1h" in payload["knots"]
        k = payload["knots"]["1h"]
        assert list(k["y"]) == sorted(k["y"])          # isotonic => monotone
        cal = apply_calibration("1h", 0.8)
        assert cal is not None and 0.0 <= cal <= 1.0
        # high-conf bucket calibrates above low-conf bucket
        assert apply_calibration("1h", 0.8) >= apply_calibration("1h", 0.45)
        store.close()

    def test_identity_when_missing(self, artifacts):
        from jobs.nightly import apply_calibration
        assert apply_calibration("1w", 0.7) is None


class TestTrainServeSkew:
    def test_db_row_and_live_dict_produce_identical_vectors(self, tmp_path):
        """The exact same prediction, read back from SQLite vs built live in
        cycle, must produce the same feature vector."""
        from persistence import Store
        store = Store(str(tmp_path / "sk.db"))
        decision = {
            "chartName": "ETHUSDT", "timeframe": "4h",
            "final": {"action": "buy", "confidence": 0.72, "score": 0.4},
            "agents": {
                "indicator": {"action": "buy", "confidence": 0.7,
                              "raw": {"action": "buy",
                                      "details": {"regime": "trend_up",
                                                  "regime_feats": {"adx": 28.0, "chop": 40.0,
                                                                   "vol_pct": 0.55, "atr": 12.5,
                                                                   "vol_ok": True},
                                                  "blend": {}}}},
                "research": {"action": "buy", "confidence": 0.6, "raw": {"action": "buy"}},
                "news": {"action": "skip", "confidence": 0.5, "raw": {"action": "SKIP"}},
                "derivatives": {"action": "sell", "confidence": 0.55,
                                "raw": {"action": "sell", "available": True,
                                        "rl": {"feats": [0.1, -0.3, -1.0, 0.2, 0.5, 0.1, 0.0, 0.2],
                                               "action_idx": 0}}},
            },
            "policy": {"weights": {}},
        }
        live_row = {
            "regime": "trend_up",
            "regime_feats": {"adx": 28.0, "chop": 40.0, "vol_pct": 0.55, "atr": 12.5, "vol_ok": True},
            "atr": 12.5, "entry_price": 3000.0, "tf": "4h", "candle_close_ts": 1751500800,
            "emitted": True, "trigger_source": "trend_continuation",
            "final_action": "buy", "final_confidence": 0.72,
            "indicator_action": "buy", "indicator_conf": 0.7,
            "research_action": "buy", "research_conf": 0.6,
            "news_action": "SKIP", "news_conf": 0.5,
            "deriv_action": "sell", "deriv_conf": 0.55,
            "deriv_feats": [0.1, -0.3, -1.0, 0.2, 0.5, 0.1, 0.0, 0.2],
        }
        pid = store.record_prediction(decision, candle_close_ts=1751500800, entry_price=3000.0,
                                      horizon_k=2, atr=12.5, tp_price=3018.75, sl_price=2987.5,
                                      trigger_source="trend_continuation", emitted=True)
        db_row = store.get_prediction(pid)
        v_live = meta_features_from_prediction_row(live_row)
        v_db = meta_features_from_prediction_row(db_row)
        assert v_live == v_db, [
            (n, a, b) for n, a, b in zip(FEATURE_NAMES, v_live, v_db) if a != b]
        store.close()


class TestNightlyCatchup:
    """v3.7: last-success marker + missed-02:00 recovery."""

    def test_training_writes_marker(self, tmp_path, artifacts):
        from persistence import Store
        from jobs.nightly import run_nightly_training
        store = Store(str(tmp_path / "m.db"))
        _insert_rows(store, n=60)          # below META_MIN_ROWS is fine
        run_nightly_training(store)
        marker = json.loads(Path(config.NIGHTLY_MARKER_PATH).read_text())
        assert marker["last_success_ts"] > 0 and marker["rows"] == 60
        store.close()

    def test_needs_catchup_truth_table(self, tmp_path):
        from datetime import datetime, timedelta
        from zoneinfo import ZoneInfo
        from jobs.nightly import _needs_catchup
        IST = ZoneInfo("Asia/Kolkata")
        now = datetime(2026, 7, 24, 10, 0, tzinfo=IST)   # boundary = today 02:00
        boundary = now.replace(hour=2, minute=0, second=0, microsecond=0)

        # missing marker -> catch up
        assert _needs_catchup(now=now, hour=2) is True
        # stale marker (before today's 02:00) -> catch up
        Path(config.NIGHTLY_MARKER_PATH).write_text(json.dumps(
            {"last_success_ts": (boundary - timedelta(hours=3)).timestamp()}))
        assert _needs_catchup(now=now, hour=2) is True
        # fresh marker (after the boundary) -> no
        Path(config.NIGHTLY_MARKER_PATH).write_text(json.dumps(
            {"last_success_ts": (boundary + timedelta(minutes=5)).timestamp()}))
        assert _needs_catchup(now=now, hour=2) is False
        # before today's boundary the reference is YESTERDAY's 02:00
        early = now.replace(hour=1)
        assert _needs_catchup(now=early, hour=2) is False
        Path(config.NIGHTLY_MARKER_PATH).write_text(json.dumps(
            {"last_success_ts": (boundary - timedelta(days=1, hours=1)).timestamp()}))
        assert _needs_catchup(now=early, hour=2) is True
