"""v3.5 sentiment voter integration: brain 5-voter math + flag parity +
legacy-policy absorption, persistence snapshot roundtrip + migration, grader
reward + manual-correction netting. All agents mocked — no network."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config

SENT_OUT = {"action": "buy", "confidence": 0.7, "available": True,
            "rl": {"feats": [0.1] * 10, "action_idx": 2}}


class TestBrainFiveVoters:
    def _dm(self, tmp_path, monkeypatch):
        import brain.decision_maker as bdm
        monkeypatch.setattr(bdm, "POLICY_PATH", str(tmp_path / "brain.json"))
        dm = bdm.DecisionMaker()
        monkeypatch.setattr(dm.indicator, "decide",
                            lambda s, tf, **k: {"action": "buy", "confidence": 0.8})
        monkeypatch.setattr(dm.research, "decide",
                            lambda *a, **k: {"action": "buy", "confidence": 0.6})
        monkeypatch.setattr(dm.news, "run", lambda *a, **k: {"action": "SKIP", "confidence": 0.5})
        monkeypatch.setattr(dm.derivatives, "decide",
                            lambda s, tf: {"action": "skip", "confidence": 0.0,
                                           "available": False, "rl": None})
        monkeypatch.setattr(dm.sentiment, "decide",
                            lambda s, tf, **k: dict(SENT_OUT))
        return dm

    @staticmethod
    def _wsum(w, with_sentiment):
        # v3.7: decide() renormalizes over the ACTIVE roster only
        active = ["indicator", "research", "news"]
        if config.DERIVATIVES_ENABLED:
            active.append("derivatives")
        if with_sentiment:
            active.append("sentiment")
        return sum(w[a] for a in active)

    def test_flag_on_includes_sentiment_vote(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "SENTIMENT_ENABLED", True)
        dm = self._dm(tmp_path, monkeypatch)
        res = dm.decide("BTCUSDT", "4h")
        w = res["policy"]["weights"]
        expected = (w["indicator"] * 1 * 0.8 + w["research"] * 1 * 0.6
                    + w["sentiment"] * 1 * 0.7) / self._wsum(w, True)
        assert res["final"]["score"] == pytest.approx(expected, abs=1e-6)
        assert res["agents"]["sentiment"]["action"] == "buy"

    def test_flag_off_sentiment_is_noop(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "SENTIMENT_ENABLED", False)
        dm = self._dm(tmp_path, monkeypatch)
        res = dm.decide("BTCUSDT", "4h")
        assert res["agents"]["sentiment"]["confidence"] == 0.0
        w = res["policy"]["weights"]
        expected = (w["indicator"] * 1 * 0.8
                    + w["research"] * 1 * 0.6) / self._wsum(w, False)
        assert res["final"]["score"] == pytest.approx(expected, abs=1e-6)

    def test_legacy_policy_absorbs_sentiment(self, tmp_path, monkeypatch):
        import brain.decision_maker as bdm
        path = tmp_path / "brain.json"
        path.write_text(json.dumps({"scores": {"indicator": 3.0, "research": 2.0,
                                               "news": 1.0, "derivatives": 1.5},
                                    "weights": None}))
        monkeypatch.setattr(bdm, "POLICY_PATH", str(path))
        dm = bdm.DecisionMaker()
        assert "sentiment" in dm.policy["weights"]
        assert abs(sum(dm.policy["weights"].values()) - 1.0) < 1e-9


def _decision_with_sentiment():
    return {
        "chartName": "BTCUSDT", "timeframe": "4h",
        "agents": {
            "indicator": {"action": "buy", "confidence": 0.8,
                          "raw": {"action": "buy",
                                  "details": {"blend": {"type1_share": 0.6}}}},
            "sentiment": {"action": "buy", "confidence": 0.7,
                          "raw": dict(SENT_OUT)},
        },
        "final": {"action": "buy", "confidence": 0.75, "score": 0.5},
        "policy": {"weights": {"indicator": 0.6, "sentiment": 0.4}},
    }


class TestPersistence:
    def test_snapshot_roundtrip(self, tmp_path):
        from persistence import Store
        store = Store(str(tmp_path / "p.db"))
        pid = store.record_prediction(_decision_with_sentiment(), candle_close_ts=1.0,
                                      entry_price=100.0, horizon_k=2, grade_due_ts=2.0)
        p = store.get_prediction(pid)
        assert p["sentiment_action"] == "buy"
        assert p["sentiment_action_idx"] == 2
        assert p["sentiment_feats"] == [0.1] * 10          # JSON col deserialized
        assert p["sentiment_conf"] == pytest.approx(0.7)
        store.close()

    def test_row_without_sentiment_is_null(self, tmp_path):
        from persistence import Store
        store = Store(str(tmp_path / "p.db"))
        d = _decision_with_sentiment()
        del d["agents"]["sentiment"]
        pid = store.record_prediction(d, candle_close_ts=1.0, entry_price=100.0,
                                      horizon_k=2, grade_due_ts=2.0)
        p = store.get_prediction(pid)
        assert p["sentiment_action"] is None and p["sentiment_feats"] is None
        store.close()


class FakeSentimentAgent:
    def __init__(self):
        self.calls = []

    def apply_reward(self, feats, action_idx, reward):
        self.calls.append((feats, action_idx, reward))


class TestGraderRewards:
    def _grader(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "TB_GRADING_ENABLED", True)
        from grader import Grader
        from persistence import Store
        from types import SimpleNamespace
        store = Store(str(tmp_path / "g.db"))
        fake = FakeSentimentAgent()
        dm = SimpleNamespace(sentiment=fake, derivatives=None,
                             news=SimpleNamespace(apply_reward=lambda *a: None),
                             research=SimpleNamespace(apply_reward=lambda *a: None),
                             indicator=SimpleNamespace(apply_reward=lambda *a: None),
                             apply_brain_feedback=lambda *a: None)
        return Grader(dm, data_fetcher=None, store=store), store, fake

    def test_auto_reward_uses_stored_snapshot(self, tmp_path, monkeypatch):
        grader, store, fake = self._grader(tmp_path, monkeypatch)
        pid = store.record_prediction(_decision_with_sentiment(), candle_close_ts=1.0,
                                      entry_price=100.0, horizon_k=2, grade_due_ts=2.0)
        assert store.claim_grading(pid, "auto")
        grader._apply_rewards(store.get_prediction(pid), "buy", source="auto")
        assert fake.calls == [([0.1] * 10, 2, config.REWARD_CORRECT)]
        rewards = {r["agent"]: r["reward"] for r in store.rewards_for(pid)}
        assert rewards["sentiment"] == config.REWARD_CORRECT
        store.close()

    def test_manual_correction_nets_prior_auto(self, tmp_path, monkeypatch):
        grader, store, fake = self._grader(tmp_path, monkeypatch)
        pid = store.record_prediction(_decision_with_sentiment(), candle_close_ts=1.0,
                                      entry_price=100.0, horizon_k=2, grade_due_ts=2.0)
        assert store.claim_grading(pid, "auto")
        grader._apply_rewards(store.get_prediction(pid), "buy", source="auto")
        out = grader.apply_manual_feedback(pid, "sell")     # human flips it
        assert out["status"] == "corrected"
        # net = wrong - correct, applied on top of the earlier auto reward
        net = config.REWARD_WRONG - config.REWARD_CORRECT
        assert fake.calls[-1] == ([0.1] * 10, 2, pytest.approx(net))
        store.close()
