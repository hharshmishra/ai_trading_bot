"""Phase 4: DerivativesAgent features, RL, brain 4-voter math. All mocked — no network."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from agents.derivatives_agent import (DerivativesAgent, DerivativesRL,
                                      build_features, deriv_note)


def snap(funding=0.0001, hist=None, oi=0.02, top=1.2, acct=1.1):
    return {"funding_rate": funding, "funding_hist": hist or [0.0001] * 10,
            "oi_change_pct": oi, "top_position_ratio": top, "global_account_ratio": acct}


class TestFeatures:
    def test_bounds(self):
        feats = build_features(snap(funding=0.01, oi=5.0, top=9.0, acct=0.01), 0.5)
        assert len(feats) == 8
        assert all(-1.0 <= f <= 1.0 for f in feats)

    def test_extreme_positive_funding_flags_short_squeeze_of_longs(self):
        feats = build_features(snap(funding=0.001), 0.0)   # 0.1%/8h >> extreme
        assert feats[2] == -1.0    # crowded longs -> bearish squeeze feature

    def test_extreme_negative_funding_flags_long_squeeze_of_shorts(self):
        feats = build_features(snap(funding=-0.001), 0.0)
        assert feats[2] == 1.0

    def test_no_extreme_below_threshold(self):
        feats = build_features(snap(funding=0.0001), 0.0)
        assert feats[2] == 0.0

    def test_oi_price_divergence_signs(self):
        up_up = build_features(snap(oi=0.05), price_change_6h=0.02)[4]
        up_down = build_features(snap(oi=0.05), price_change_6h=-0.02)[4]
        down_up = build_features(snap(oi=-0.05), price_change_6h=0.02)[4]
        down_down = build_features(snap(oi=-0.05), price_change_6h=-0.02)[4]
        assert up_up == 0.5 and up_down == -0.5
        assert down_up == -0.25 and down_down == 0.25

    def test_skew_signs(self):
        long_heavy = build_features(snap(top=2.0), 0.0)
        short_heavy = build_features(snap(top=0.5), 0.0)
        assert long_heavy[5] > 0 > short_heavy[5]


class TestRL:
    def test_update_moves_weights_and_persists(self, tmp_path):
        path = str(tmp_path / "pol.json")
        rl = DerivativesRL(policy_path=path)
        feats = [0.5, 0.1, -1.0, 0.3, 0.5, 0.2, 0.1, 0.2]
        before = [row[:] for row in rl.weights]
        rl.update(feats, 2, 1.0)   # reward the buy action
        assert rl.weights != before
        rl2 = DerivativesRL(policy_path=path)     # reload from disk
        assert rl2.weights == rl.weights

    def test_reward_direction(self, tmp_path):
        rl = DerivativesRL(policy_path=str(tmp_path / "p.json"))
        feats = [1.0] * 8
        p_before = rl.prob(feats, 2)
        rl.update(feats, 2, 4.0)
        assert rl.prob(feats, 2) > p_before      # rewarded action gains probability


class TestAgent:
    def _agent(self, tmp_path, monkeypatch, snapshot):
        import agents.derivatives_agent as da
        monkeypatch.setattr(da.dfx, "fetch_derivatives", lambda s: snapshot)
        class _F:
            def get_ohlcv(self, *a, **k):
                import pandas as pd
                import numpy as np
                n = 10
                return pd.DataFrame({"timestamp": pd.date_range("2026-01-01", periods=n, freq="h"),
                                     "open": np.full(n, 100.0), "high": np.full(n, 101.0),
                                     "low": np.full(n, 99.0), "close": np.linspace(100, 103, n),
                                     "volume": np.ones(n)})
        agent = DerivativesAgent(data_fetcher=_F())
        agent._rl = DerivativesRL(policy_path=str(tmp_path / "pol.json"))
        agent._rl.epsilon = 0.0    # deterministic in tests
        return agent

    def test_available_path(self, tmp_path, monkeypatch):
        agent = self._agent(tmp_path, monkeypatch, snap())
        out = agent.decide("BTCUSDT", "4h")
        assert out["available"] is True
        assert out["action"] in ("buy", "sell", "skip")
        assert 0.5 <= out["confidence"] <= 0.9
        assert len(out["rl"]["feats"]) == 8

    def test_unavailable_is_noop_shape(self, tmp_path, monkeypatch):
        agent = self._agent(tmp_path, monkeypatch, None)
        out = agent.decide("LUNAUSDT", "4h")
        assert out == {"agent": "derivatives_agent", "chartName": "LUNAUSDT",
                       "timeframe": "4h", "action": "skip", "confidence": 0.0,
                       "available": False, "rl": None}

    def test_apply_reward_none_feats_is_noop(self, tmp_path, monkeypatch):
        agent = self._agent(tmp_path, monkeypatch, snap())
        before = [row[:] for row in agent._rl.weights]
        agent.apply_reward(None, None, 1.0)
        assert agent._rl.weights == before


class TestBrainFourVoters:
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
                            lambda s, tf: {"action": "sell", "confidence": 0.7,
                                           "available": True, "rl": {"feats": [0.1] * 8, "action_idx": 0}})
        return dm

    def test_flag_on_includes_derivatives_vote(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "DERIVATIVES_ENABLED", True)
        dm = self._dm(tmp_path, monkeypatch)
        res = dm.decide("BTCUSDT", "4h")
        w = res["policy"]["weights"]
        expected = (w["indicator"] * 1 * 0.8 + w["research"] * 1 * 0.6
                    + w["news"] * 0 * 0.5 + w["derivatives"] * -1 * 0.7)
        assert res["final"]["score"] == pytest.approx(expected, abs=1e-6)
        assert res["agents"]["derivatives"]["action"] == "sell"

    def test_flag_off_derivatives_is_noop(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "DERIVATIVES_ENABLED", False)
        dm = self._dm(tmp_path, monkeypatch)
        res = dm.decide("BTCUSDT", "4h")
        assert res["agents"]["derivatives"]["confidence"] == 0.0
        w = res["policy"]["weights"]
        expected = w["indicator"] * 1 * 0.8 + w["research"] * 1 * 0.6
        assert res["final"]["score"] == pytest.approx(expected, abs=1e-6)

    def test_legacy_policy_absorbs_new_voter(self, tmp_path, monkeypatch):
        import json
        import brain.decision_maker as bdm
        path = tmp_path / "brain.json"
        path.write_text(json.dumps({"scores": {"indicator": 3.0, "research": 2.0, "news": 1.0},
                                    "weights": None}))
        monkeypatch.setattr(bdm, "POLICY_PATH", str(path))
        dm = bdm.DecisionMaker()
        assert "derivatives" in dm.policy["weights"]
        assert abs(sum(dm.policy["weights"].values()) - 1.0) < 1e-9


class TestDerivNote:
    def test_note_formats(self):
        n = deriv_note({"funding_rate": 0.0008, "oi_change_pct": 0.03})
        assert "funding +0.080%/8h" in n and "OI rising" in n and "squeeze risk" in n
        assert deriv_note(None) is None
