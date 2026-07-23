"""v3.7 learning repair: softmax trust weights (clamp/floor/self-heal),
decide-time active-roster renormalization, symmetric brain trust rewards,
bandit weight clamps. Prod evidence: 19-day run drove indicator trust to
2.3e-06 and crowned the never-voting sentiment agent (shift-normalize defect),
while news bandit weights exploded to ±287."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config


def _dm(tmp_path, monkeypatch):
    import brain.decision_maker as bdm
    monkeypatch.setattr(bdm, "POLICY_PATH", str(tmp_path / "brain.json"))
    return bdm.DecisionMaker()


# ----------------------------- trust weights ------------------------------- #
class TestSoftmaxTrust:
    def test_default_scores_give_expected_weights(self, tmp_path, monkeypatch):
        dm = _dm(tmp_path, monkeypatch)
        w = dm.policy["weights"]
        assert w["indicator"] == pytest.approx(0.3426, abs=2e-3)
        assert w["research"] == pytest.approx(0.2078, abs=2e-3)
        assert w["news"] == pytest.approx(0.1260, abs=2e-3)
        assert w["derivatives"] == pytest.approx(0.1618, abs=2e-3)
        assert w["sentiment"] == pytest.approx(0.1618, abs=2e-3)
        assert sum(w.values()) == pytest.approx(1.0)

    def test_legacy_exploded_scores_self_heal(self, tmp_path, monkeypatch):
        """The prod v3.6 policy file (indicator -1525) must load into clamped
        scores and finite weights — nobody stays mathematically unrecoverable."""
        import brain.decision_maker as bdm
        path = tmp_path / "brain.json"
        path.write_text(json.dumps({"scores": {
            "indicator": -1524.98, "research": -1057.17, "news": -464.08,
            "derivatives": -286.05, "sentiment": 1.5}, "weights": None}))
        monkeypatch.setattr(bdm, "POLICY_PATH", str(path))
        dm = bdm.DecisionMaker()
        assert all(-10.0 <= s <= 10.0 for s in dm.policy["scores"].values())
        w = dm.policy["weights"]
        assert sum(w.values()) == pytest.approx(1.0)
        assert all(v >= bdm.WEIGHT_FLOOR - 1e-12 for v in w.values())

    def test_floor_is_exact_two_pass(self, tmp_path, monkeypatch):
        """One agent at +10, four at -10: the four get EXACTLY the floor and
        the leader gets the remaining mass."""
        import brain.decision_maker as bdm
        path = tmp_path / "brain.json"
        path.write_text(json.dumps({"scores": {
            "indicator": 10.0, "research": -10.0, "news": -10.0,
            "derivatives": -10.0, "sentiment": -10.0}, "weights": None}))
        monkeypatch.setattr(bdm, "POLICY_PATH", str(path))
        dm = bdm.DecisionMaker()
        w = dm.policy["weights"]
        for ag in ("research", "news", "derivatives", "sentiment"):
            assert w[ag] == pytest.approx(bdm.WEIGHT_FLOOR)
        assert w["indicator"] == pytest.approx(1.0 - 4 * bdm.WEIGHT_FLOOR)
        assert sum(w.values()) == pytest.approx(1.0)


# ------------------------ decide-time renormalization ---------------------- #
class TestActiveRosterRenorm:
    def test_use_agents_subset_renormalizes(self, tmp_path, monkeypatch):
        dm = _dm(tmp_path, monkeypatch)
        monkeypatch.setattr(dm.indicator, "decide",
                            lambda s, tf, **k: {"action": "buy", "confidence": 0.8})
        monkeypatch.setattr(dm.news, "run",
                            lambda *a, **k: {"action": "BUY", "confidence": 0.5})
        monkeypatch.setattr(dm, "_headlines_for", lambda s: None)
        res = dm.decide("BTCUSDT", "4h", use_agents=("indicator", "news"))
        w = res["policy"]["weights"]
        wsum = w["indicator"] + w["news"]
        expected = (w["indicator"] * 0.8 + w["news"] * 0.5) / wsum
        assert res["final"]["score"] == pytest.approx(expected, abs=1e-6)
        # confidence is the same ratio with or without renormalization
        assert res["final"]["confidence"] == pytest.approx(1.0, abs=1e-4)

    def test_disabled_agent_holds_no_vote_mass(self, tmp_path, monkeypatch):
        """With sentiment disabled, the score must renormalize over the active
        roster — a dead agent must not shrink live votes vs the ±0.05 deadzone."""
        monkeypatch.setattr(config, "SENTIMENT_ENABLED", False)
        monkeypatch.setattr(config, "DERIVATIVES_ENABLED", False)
        dm = _dm(tmp_path, monkeypatch)
        monkeypatch.setattr(dm.indicator, "decide",
                            lambda s, tf, **k: {"action": "buy", "confidence": 0.3})
        monkeypatch.setattr(dm.research, "decide",
                            lambda *a, **k: {"action": "skip", "confidence": 0.0})
        monkeypatch.setattr(dm.news, "run",
                            lambda *a, **k: {"action": "SKIP", "confidence": 0.0})
        monkeypatch.setattr(dm, "_headlines_for", lambda s: None)
        res = dm.decide("BTCUSDT", "4h")
        w = res["policy"]["weights"]
        wsum = w["indicator"] + w["research"] + w["news"]
        assert res["final"]["score"] == pytest.approx(w["indicator"] * 0.3 / wsum,
                                                      abs=1e-6)


# --------------------------- symmetric trust map --------------------------- #
class TestBrainTrustDelta:
    def test_map_values(self):
        from brain.decision_maker import brain_trust_delta
        assert brain_trust_delta("buy", "buy", 0.8) == pytest.approx(0.8)
        assert brain_trust_delta("buy", "sell", 0.8) == pytest.approx(-0.8)
        assert brain_trust_delta("sell", "skip", 0.8) == pytest.approx(-0.2)
        assert brain_trust_delta("skip", "buy", 0.9) == 0.0
        assert brain_trust_delta("skip", "skip", 0.9) == 0.0

    def test_scores_clamp_at_rails(self, tmp_path, monkeypatch):
        import brain.decision_maker as bdm
        dm = _dm(tmp_path, monkeypatch)
        dm.policy["scores"]["indicator"] = 9.999
        for _ in range(5):
            dm.apply_brain_feedback(
                {"indicator": {"action": "buy", "confidence": 1.0}}, "buy")
        assert dm.policy["scores"]["indicator"] == pytest.approx(bdm.SCORE_CLAMP)
        dm.policy["scores"]["indicator"] = -9.999
        for _ in range(5):
            dm.apply_brain_feedback(
                {"indicator": {"action": "buy", "confidence": 1.0}}, "sell")
        assert dm.policy["scores"]["indicator"] == pytest.approx(-bdm.SCORE_CLAMP)


# ----------------------------- bandit clamps ------------------------------- #
class TestBanditWeightClamp:
    """Prod news weights hit +-287 (lr 0.1, no bound); all four bandits now
    clamp per-weight to +-5 inside update()."""

    def _drive(self, rl, feats, action, n=60):
        for _ in range(n):
            rl.update(list(feats), action, -4.0)
            rl.update(list(feats), action, -4.0)

    def test_news_clamped_and_lr_aligned(self, tmp_path, monkeypatch):
        import agents.news_agent as na
        monkeypatch.setattr(na, "POLICY_PATH", str(tmp_path / "n.json"))
        rl = na.NewsRL()
        assert rl.lr == pytest.approx(0.05)
        rl.policy.weights[0][0] = 4.95
        self._drive(rl, [1.0] * 10, 0)
        assert all(abs(w) <= na.WEIGHT_CLAMP + 1e-9
                   for row in rl.policy.weights for w in row)

    def test_research_clamped(self, tmp_path, monkeypatch):
        import agents.research_agent as ra
        monkeypatch.setattr(ra, "POLICY_PATH", str(tmp_path / "r.json"))
        rl = ra.ResearchRL(10)
        self._drive(rl, [1.0] * rl.n_features, 0)
        assert all(abs(w) <= ra.WEIGHT_CLAMP + 1e-9
                   for row in rl.policy.weights for w in row)

    def test_derivatives_clamped(self, tmp_path, monkeypatch):
        import agents.derivatives_agent as da
        monkeypatch.setattr(da, "POLICY_PATH", str(tmp_path / "d.json"))
        rl = da.DerivativesRL(policy_path=str(tmp_path / "d.json"))
        self._drive(rl, [1.0] * 8, 0)
        assert all(abs(w) <= da.WEIGHT_CLAMP + 1e-9
                   for row in rl.weights for w in row)

    def test_sentiment_clamped(self, tmp_path):
        import agents.sentiment_agent as sa
        rl = sa.SentimentRL(policy_path=str(tmp_path / "s.json"))
        self._drive(rl, [1.0] * 10, 0)
        assert all(abs(w) <= sa.WEIGHT_CLAMP + 1e-9
                   for row in rl.weights for w in row)
