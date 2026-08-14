"""v3.8 learning repair II — meta train/serve skew fix, sent-direction TB
grading, advantage-baseline trust, nightly decay, candidate telemetry.

Evidence base: 21-day prod audit (Jul 24 - Aug 14 2026). The meta model served
meta_p 0.97-1.0 on a ~37%-hit cohort because `emitted`/`trigger_source` were
leakage features whose train values differed from serve values; emitted rows
were TB-graded on the brain final while Telegram carried the trigger's side;
symmetric trust rewards sank every score (derivatives pinned at -10).
"""
import asyncio
import json
from types import SimpleNamespace

import pandas as pd
import pytest

from tests.test_phase4_runtime import make_decision


# ------------------------- meta features v2 -------------------------------- #
class TestMetaFeaturesV2:
    def test_emitted_is_not_a_feature(self):
        from jobs.features import FEATURE_NAMES
        assert "emitted" not in FEATURE_NAMES

    def test_new_features_present(self):
        from jobs.features import FEATURE_NAMES
        for name in ("trigger_sms", "candidate_side", "vol_x_nwe",
                     "sms_strength", "sms_confidence", "cvd_norm"):
            assert name in FEATURE_NAMES

    def test_vector_length_matches_names(self):
        from jobs.features import FEATURE_NAMES, meta_features_from_prediction_row
        v = meta_features_from_prediction_row({})
        assert len(v) == len(FEATURE_NAMES)

    def test_candidate_trigger_sets_one_hot(self):
        from jobs.features import FEATURE_NAMES, meta_features_from_prediction_row
        v = meta_features_from_prediction_row({"candidate_trigger": "nwe"})
        assert v[FEATURE_NAMES.index("trigger_nwe")] == 1.0
        v = meta_features_from_prediction_row({"candidate_trigger": "sms"})
        assert v[FEATURE_NAMES.index("trigger_sms")] == 1.0

    def test_trigger_source_fallback_for_legacy_rows(self):
        """Pre-v3.8 training rows only have trigger_source (a reason string)."""
        from jobs.features import FEATURE_NAMES, meta_features_from_prediction_row
        v = meta_features_from_prediction_row({"trigger_source": "nwe_ranging"})
        assert v[FEATURE_NAMES.index("trigger_nwe")] == 1.0

    def test_candidate_side_encoding(self):
        from jobs.features import FEATURE_NAMES, meta_features_from_prediction_row
        i = FEATURE_NAMES.index("candidate_side")
        assert meta_features_from_prediction_row({"candidate_action": "buy"})[i] == 1.0
        assert meta_features_from_prediction_row({"candidate_action": "sell"})[i] == 0.0
        assert meta_features_from_prediction_row({})[i] == 0.5

    def test_vol_x_nwe_interaction(self):
        from jobs.features import FEATURE_NAMES, meta_features_from_prediction_row
        i = FEATURE_NAMES.index("vol_x_nwe")
        row = {"candidate_trigger": "nwe", "regime_feats": {"vol_pct": 0.8}}
        assert meta_features_from_prediction_row(row)[i] == pytest.approx(0.8)
        row["candidate_trigger"] = "trend"
        assert meta_features_from_prediction_row(row)[i] == 0.0

    def test_sms_metrics_from_regime_feats(self):
        from jobs.features import FEATURE_NAMES, meta_features_from_prediction_row
        rf = {"sms_strength": -71.0, "sms_conf": 75.0, "cvd_norm": -0.4}
        v = meta_features_from_prediction_row({"regime_feats": rf})
        assert v[FEATURE_NAMES.index("sms_strength")] == pytest.approx(-0.71)
        assert v[FEATURE_NAMES.index("sms_confidence")] == pytest.approx(0.75)
        assert v[FEATURE_NAMES.index("cvd_norm")] == pytest.approx(-0.4)

    def test_train_equals_serve_vector(self):
        """THE skew regression: the dict cycle builds at decide time and the
        DB row nightly training reads must produce the identical vector —
        including for a candidate that later got suppressed (the v3.7 bug:
        serve said emitted=1/trigger=nwe, train said emitted=0/trigger=None).
        """
        from jobs.features import meta_features_from_prediction_row
        shared = {
            "regime": "mixed", "regime_feats": {"adx": 20.0, "chop": 50.0,
                                                "vol_pct": 0.7, "vol_ok": True},
            "atr": 1.0, "entry_price": 100.0, "tf": "1h",
            "candle_close_ts": 1700000000.0,
            "candidate_trigger": "nwe", "candidate_action": "sell",
            "final_action": "buy", "final_confidence": 0.4,
            "indicator_action": "buy", "indicator_conf": 0.9,
            "research_action": "sell", "research_conf": 0.6,
            "news_action": "buy", "news_conf": 0.5,
            "deriv_action": None, "deriv_conf": None, "deriv_feats": None,
        }
        serve = dict(shared)                          # cycle's row_like
        train = dict(shared, emitted=0, trigger_source=None,
                     realized_label="sell")           # suppressed row from DB
        assert meta_features_from_prediction_row(serve) == \
               meta_features_from_prediction_row(train)


# ------------------------- candidate derivation ---------------------------- #
class TestDeriveCandidate:
    def _block(self, nwe=None, trend=None):
        direct = []
        if nwe:
            direct.append({"name": "nwe", "signal": nwe})
        if trend:
            direct.append({"name": "supertrend_flip", "signal": trend})
        return {"raw": {"details": {"direct_signals": direct}}}

    def test_no_candidate(self):
        from signals import derive_candidate
        assert derive_candidate("", self._block(), "buy") == (None, None)

    def test_nwe_reasons(self):
        from signals import derive_candidate
        blk = self._block(nwe="sell")
        for reason in ("nwe_ranging", "nwe_mixed", "nwe_mixed_disabled",
                       "nwe_higher_tf_disabled", "nwe_high_vol", "nwe_direct",
                       "no_brain_agreement"):
            assert derive_candidate(reason, blk, "buy") == ("nwe", "sell")

    def test_low_volume_resolves_by_nwe_presence(self):
        from signals import derive_candidate
        assert derive_candidate("low_volume", self._block(nwe="buy"), "skip") == ("nwe", "buy")
        assert derive_candidate("low_volume", self._block(trend="sell"), "skip") == ("trend", "sell")

    def test_trend_reasons(self):
        from signals import derive_candidate
        blk = self._block(trend="sell")
        for reason in ("trend_continuation", "trend_reversal", "trend_1h",
                       "reversal_disabled", "counter_trend_no_flip"):
            assert derive_candidate(reason, blk, "buy") == ("trend", "sell")

    def test_conf_reasons_use_final(self):
        from signals import derive_candidate
        blk = self._block()
        assert derive_candidate("conf_over_80", blk, "buy") == ("conf", "buy")
        assert derive_candidate("counter_trend_conf", blk, "sell") == ("conf", "sell")
        assert derive_candidate("conf_over_80", blk, "skip") == ("conf", None)

    def test_sms_reason(self):
        from signals import derive_candidate
        blk = {"raw": {"details": {"direct_signals": [
            {"name": "sms_bos", "signal": "buy"}]}}}
        assert derive_candidate("sms_bos", blk, "skip") == ("sms", "buy")


# --------------------------- trust advantage ------------------------------- #
class TestTrustAdvantage:
    def test_outcome_scores(self):
        from brain.decision_maker import brain_trust_outcome
        assert brain_trust_outcome("buy", "buy") == 1.0
        assert brain_trust_outcome("buy", "sell") == -1.0
        assert brain_trust_outcome("sell", "skip") == -0.25
        assert brain_trust_outcome("skip", "buy") is None

    def test_delta_backcompat_zero_baseline(self):
        """v3.7 call shape still returns the symmetric values."""
        from brain.decision_maker import brain_trust_delta
        assert brain_trust_delta("buy", "buy", 0.8) == pytest.approx(0.8)
        assert brain_trust_delta("buy", "sell", 0.8) == pytest.approx(-0.8)
        assert brain_trust_delta("buy", "skip", 0.8) == pytest.approx(-0.2)
        assert brain_trust_delta("skip", "buy", 0.8) == 0.0

    def test_delta_subtracts_baseline(self):
        from brain.decision_maker import brain_trust_delta
        # an agent whose recent average is already -0.11 gains for merely
        # being less bad than usual — the anti-sink property
        assert brain_trust_delta("buy", "skip", 1.0, baseline=-0.5) == pytest.approx(0.25)
        assert brain_trust_delta("buy", "buy", 1.0, baseline=0.9) == pytest.approx(0.1)

    def test_feedback_updates_baseline_after_delta(self, dm):
        from brain.decision_maker import TRUST_BASELINE_ALPHA, TRUST_LR
        res = {"indicator": {"action": "buy", "confidence": 1.0}}
        s0 = dm.policy["scores"]["indicator"]
        dm.apply_brain_feedback(res, "buy")
        # first vote judged against baseline 0 -> full +1*conf delta
        assert dm.policy["scores"]["indicator"] == pytest.approx(s0 + TRUST_LR * 1.0)
        assert dm.policy["baseline"]["indicator"] == pytest.approx(TRUST_BASELINE_ALPHA)
        # second identical vote judged against the moved baseline
        s1 = dm.policy["scores"]["indicator"]
        dm.apply_brain_feedback(res, "buy")
        expect = s1 + TRUST_LR * (1.0 - TRUST_BASELINE_ALPHA)
        assert dm.policy["scores"]["indicator"] == pytest.approx(expect)

    def test_negative_sum_environment_no_longer_sinks(self, dm):
        """At the measured prod mix (~28% correct / 29% opposite / 43% flat)
        v3.7 scores sank ~ -0.11/vote forever. With the baseline the long-run
        drift converges to ~0 instead of the -10 rail."""
        import itertools
        res = {"indicator": {"action": "buy", "confidence": 1.0}}
        outcomes = ["buy"] * 28 + ["sell"] * 29 + ["skip"] * 43
        for out in itertools.chain.from_iterable([outcomes] * 8):
            dm.apply_brain_feedback({"indicator": dict(res["indicator"])}, out)
        assert dm.policy["scores"]["indicator"] > -3.0   # v3.7 math: ~ -7.9

    def test_decay_pulls_scores_toward_zero(self, dm, monkeypatch):
        import config
        monkeypatch.setattr(config, "TRUST_DECAY", 0.5)
        dm.policy["scores"] = {k: -10.0 for k in dm.policy["scores"]}
        dm.decay_trust()
        assert all(v == pytest.approx(-5.0) for v in dm.policy["scores"].values())

    def test_decay_noop_out_of_range(self, dm):
        before = dict(dm.policy["scores"])
        dm.decay_trust(1.0)
        dm.decay_trust(0.0)
        dm.decay_trust(-2.0)
        assert dm.policy["scores"] == before

    def test_decay_and_feedback_are_serialized(self, dm, tmp_path):
        """Review finding: nightly decay_trust races the grader's feedback
        thread — both now hold _POLICY_LOCK and the save is atomic, so the
        policy file must never be torn/invalid under interleaving."""
        import json
        import threading
        import brain.decision_maker as dmm
        res = {"indicator": {"action": "buy", "confidence": 1.0}}
        errors = []

        def feedback():
            try:
                for i in range(150):
                    dm.apply_brain_feedback(
                        {"indicator": dict(res["indicator"])},
                        "buy" if i % 3 else "sell")
            except Exception as e:            # pragma: no cover
                errors.append(e)

        def decay():
            try:
                for _ in range(150):
                    dm.decay_trust(0.999)
            except Exception as e:            # pragma: no cover
                errors.append(e)

        threads = [threading.Thread(target=feedback), threading.Thread(target=decay)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        assert not errors
        with open(dmm.POLICY_PATH, encoding="utf-8") as f:
            pol = json.load(f)                # never torn — atomic replace
        assert all(abs(v) <= 10.0 for v in pol["scores"].values())

    @pytest.fixture
    def dm(self, tmp_path, monkeypatch):
        import brain.decision_maker as dmm
        monkeypatch.setattr(dmm, "POLICY_PATH", str(tmp_path / "brain_policy.json"))
        d = object.__new__(dmm.DecisionMaker)   # skip agent construction
        d.policy = dmm._load_policy()
        d._normalize_weights()
        return d


# ---------------------- grade what was sent (TB) --------------------------- #
class TestSentDirectionGrading:
    def _grade(self, tmp_path, monkeypatch, *, emitted, candidate_action,
               final_action, path_high, path_low):
        """Run one prediction through Grader._grade_prediction with a fixed
        post-entry path and return (realized_label, label_tb)."""
        import config
        from grader import Grader
        from persistence import Store

        monkeypatch.setattr(config, "TB_GRADING_ENABLED", True)
        store = Store(str(tmp_path / "g.db"))
        decision = make_decision(action=final_action, tf="1h",
                                 nwe=candidate_action if emitted else None)
        pid = store.record_prediction(
            decision, candle_close_ts=1000.0, entry_price=100.0, horizon_k=3,
            grade_due_ts=2000.0, emitted=emitted,
            tp_price=101.5 if candidate_action != "sell" or not emitted else 98.5,
            sl_price=99.0 if candidate_action != "sell" or not emitted else 101.0,
            candidate_trigger="nwe" if emitted else None,
            candidate_action=candidate_action if emitted else None)

        # pre-entry rows so the truncated-window guard (len(after)==len(df))
        # sees a window that reaches back before the entry candle
        df = pd.DataFrame({
            "timestamp": pd.to_datetime([1000.0 + i * 3600 for i in range(-16, 8)],
                                        unit="s"),
            "open": 100.0, "high": path_high, "low": path_low,
            "close": 100.0, "volume": 1.0})
        fetcher = SimpleNamespace(get_ohlcv=lambda p, tf, limit=50: df.copy())
        dm = SimpleNamespace(news=SimpleNamespace(apply_reward=lambda *a: None),
                             research=SimpleNamespace(apply_reward=lambda *a: None),
                             indicator=SimpleNamespace(apply_reward=lambda *a: None),
                             apply_brain_feedback=lambda *a, **k: None)
        g = Grader(dm, data_fetcher=fetcher, store=store)
        graded = g.grade_once(now_ts=3000.0 + 8 * 3600)
        assert len(graded) == 1
        with store._lock:
            out = store.conn.execute(
                "SELECT realized_label, label_tb FROM outcomes "
                "WHERE prediction_id = ?", (pid,)).fetchone()
        store.close()
        return out["realized_label"], out["label_tb"]

    def test_emitted_row_graded_on_sent_direction(self, tmp_path, monkeypatch):
        """Sent SELL (nwe), brain final BUY, price collapses: the SELL's TP
        (below entry) is hit -> label 'sell' (sent direction correct). The
        v3.7 code would have graded the BUY's SL and called it 'sell' too by
        accident of geometry — but with tp above entry it labeled tp='buy'."""
        label, tb = self._grade(tmp_path, monkeypatch, emitted=True,
                                candidate_action="sell", final_action="buy",
                                path_high=100.2, path_low=98.0)
        assert (label, tb) == ("sell", "tp")

    def test_emitted_sent_direction_wrong_is_sl(self, tmp_path, monkeypatch):
        """Sent SELL, price rips up through the SELL's SL -> label 'buy'."""
        label, tb = self._grade(tmp_path, monkeypatch, emitted=True,
                                candidate_action="sell", final_action="buy",
                                path_high=101.6, path_low=99.9)
        assert (label, tb) == ("buy", "sl")

    def test_legacy_shadow_row_keeps_final_action_path(self, tmp_path, monkeypatch):
        """Non-emitted directional final: unchanged v3.7 behavior."""
        label, tb = self._grade(tmp_path, monkeypatch, emitted=False,
                                candidate_action="skip", final_action="buy",
                                path_high=101.6, path_low=99.5)
        assert (label, tb) == ("buy", "tp")


# ---------------------- persistence + reset + nightly ---------------------- #
class TestCandidateTelemetry:
    def test_columns_migrated_and_recorded(self, tmp_path):
        from persistence import Store
        store = Store(str(tmp_path / "p.db"))
        pid = store.record_prediction(make_decision(), candidate_trigger="nwe",
                                      candidate_action="sell")
        row = store.get_prediction(pid)
        assert row["candidate_trigger"] == "nwe"
        assert row["candidate_action"] == "sell"
        store.close()

    def test_news_action_persisted_lowercase(self, tmp_path):
        from persistence import Store
        store = Store(str(tmp_path / "n.db"))
        pid = store.record_prediction(make_decision())      # news says "BUY"
        assert store.get_prediction(pid)["news_action"] == "buy"
        store.close()

    def test_reset_artifacts_include_marker(self):
        import config
        from scripts.reset_learning import _nightly_artifacts
        assert config.NIGHTLY_MARKER_PATH in _nightly_artifacts()

    def test_nightly_decays_trust_when_dm_passed(self, tmp_path, monkeypatch):
        import config
        from jobs.nightly import run_nightly_training
        from persistence import Store
        calls = []
        dm = SimpleNamespace(decay_trust=lambda: calls.append(True))
        store = Store(str(tmp_path / "t.db"))
        summary = run_nightly_training(store, dm)
        assert calls == [True] and summary["trust_decayed"] is True
        summary2 = run_nightly_training(store)
        assert summary2["trust_decayed"] is False
        store.close()


class TestCycleCandidatePersistence:
    def test_suppressed_candidate_keeps_trigger_and_action(self, tmp_path, monkeypatch):
        """A meta-gated NWE candidate must persist candidate_trigger='nwe' +
        its direction even though trigger_source stays NULL (v3.7 lost both)."""
        import config
        from cycle import run_cycle
        from persistence import Store

        store = Store(str(tmp_path / "c.db"))
        df = pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=10, freq="1h"),
                           "open": 1.0, "high": 1.0, "low": 1.0, "close": 100.0,
                           "volume": 1.0})
        fetcher = SimpleNamespace(get_ohlcv=lambda s, tf, limit=500: df.copy())
        decision = make_decision(pair="AUSDT", tf="1h", action="skip", conf=0.1,
                                 nwe="sell")
        decision["agents"]["indicator"]["raw"]["details"]["regime"] = "ranging"
        decision["agents"]["indicator"]["raw"]["details"]["regime_feats"] = {
            "vol_ok": True, "vol_pct": 0.5}
        dm = SimpleNamespace(indicator=None, news=None, research=None,
                             decide=lambda sym, tf, ua, ctx: decision)

        monkeypatch.setattr(config, "GATE_V2_ENABLED", True)
        monkeypatch.setattr(config, "META_GATE_ENABLED", True)
        monkeypatch.setattr(config, "META_GATE_THRESHOLD", 0.55)
        import cycle as cycle_mod
        monkeypatch.setattr(cycle_mod, "meta_probability", lambda row: 0.10)

        summary = asyncio.run(run_cycle(
            ["1h"], dm=dm, data_fetcher=fetcher, broadcast=None,
            symbols=["AUSDT"], store=store, build_context=lambda *a, **k: None))
        with store._lock:
            row = store.conn.execute(
                "SELECT gate_reason, trigger_source, candidate_trigger, "
                "candidate_action, emitted FROM predictions").fetchone()
        assert row["emitted"] == 0
        assert row["gate_reason"] == "meta_gate"
        assert row["trigger_source"] is None
        assert row["candidate_trigger"] == "nwe"
        assert row["candidate_action"] == "sell"
        store.close()
