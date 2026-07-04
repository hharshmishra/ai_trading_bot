"""v3.4 per-rule type-2 credibility learning (T2_RULE_LEARNING): fired_rules
snapshot, sign-anchored nudges with clips, weighted tally, schema migration,
and flag-off parity with v3.3."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from agents import indicator_agent as ia
from agents.indicator_agent import IndicatorAgent


def _downtrend(n=220):
    close = np.linspace(300, 100, n)
    return pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=n, freq="h"),
        "open": np.concatenate([[close[0]], close[:-1]]),
        "high": close + 0.4, "low": close - 0.4, "close": close,
        "volume": np.full(n, 1000.0),
    })


def _decide(monkeypatch, learning: bool):
    monkeypatch.setattr(config, "T2_RULE_LEARNING", learning)
    return IndicatorAgent().decide("X", "1h", ohlcv=_downtrend(), log=False)


class TestFiredRules:
    def test_flag_off_no_fired_rules_anywhere(self, monkeypatch):
        d = _decide(monkeypatch, False)
        assert "fired_rules" not in d.details["type2"]
        assert "fired_rules" not in d.details["blend"]

    def test_flag_on_records_supporters_of_type2_action(self, monkeypatch):
        d = _decide(monkeypatch, True)
        t2 = d.details["type2"]
        assert t2["action"] == "sell"                     # hard downtrend
        fired = t2["fired_rules"]
        assert "ribbon" in fired and "supertrend" in fired and "macd" in fired
        assert "rsi14" not in fired                       # oversold votes BULL here
        assert d.details["blend"]["fired_rules"] == fired

    def test_default_weights_keep_v33_action_and_confidence(self, monkeypatch):
        off = _decide(monkeypatch, False)
        on = _decide(monkeypatch, True)                   # all weights 1.0
        assert on.details["type2"]["action"] == off.details["type2"]["action"]
        assert on.details["type2"]["confidence"] == pytest.approx(
            off.details["type2"]["confidence"])


class TestApplyReward:
    def _blend(self, fired):
        return {"type1_share": 0.5, "type2_share": 0.5, "fired_rules": fired}

    def test_win_and_wrong_anchor_steps(self, monkeypatch):
        monkeypatch.setattr(config, "T2_RULE_LEARNING", True)
        ag = IndicatorAgent()
        ag.apply_reward(self._blend(["ribbon", "macd"]), 1.0)
        w = ag.policy["type2_rules"]
        assert w["ribbon"]["weight"] == pytest.approx(1.05)     # +0.05 win
        assert w["macd"]["weight"] == pytest.approx(1.05)
        ag.apply_reward(self._blend(["ribbon"]), -4.0)
        assert ag.policy["type2_rules"]["ribbon"]["weight"] == pytest.approx(1.05 - 0.07)
        assert ag.policy["type2_rules"]["macd"]["weight"] == pytest.approx(1.05)  # untouched

    def test_clips_hold(self, monkeypatch):
        monkeypatch.setattr(config, "T2_RULE_LEARNING", True)
        ag = IndicatorAgent()
        for _ in range(40):
            ag.apply_reward(self._blend(["bb"]), -4.0)
        assert ag.policy["type2_rules"]["bb"]["weight"] == pytest.approx(0.1)   # floor
        for _ in range(60):
            ag.apply_reward(self._blend(["bb"]), 1.0)
        assert ag.policy["type2_rules"]["bb"]["weight"] == pytest.approx(2.0)   # cap

    def test_flag_off_leaves_policy_rules_empty(self, monkeypatch):
        monkeypatch.setattr(config, "T2_RULE_LEARNING", False)
        ag = IndicatorAgent()
        ag.apply_reward({"type1_share": 0.5, "type2_share": 0.5}, 1.0)
        assert ag.policy["type2_rules"] == {}


class TestWeightedTally:
    def test_learned_weight_changes_the_tally(self, monkeypatch):
        monkeypatch.setattr(config, "T2_RULE_LEARNING", True)
        ag = IndicatorAgent()
        base = ag.decide("X", "1h", ohlcv=_downtrend(), log=False)
        # crush the credibility of every bear rule that fired
        pol = dict(ag.policy)
        pol["type2_rules"] = {k: {"weight": 0.1, "score": 0}
                              for k in base.details["type2"]["fired_rules"]}
        ag.policy = pol
        crushed = ag.decide("X", "1h", ohlcv=_downtrend(), log=False)
        assert (crushed.details["type2"]["votes"]["bear"]
                < base.details["type2"]["votes"]["bear"])


class TestMigration:
    def test_old_policy_gains_type2_rules_on_load(self, tmp_path, monkeypatch):
        p = tmp_path / "pol.json"
        p.write_text(json.dumps({"weights": {"type1": 0.6, "type2": 0.4},
                                 "direct_signals": {}, "score": 3}))
        monkeypatch.setattr(ia, "POLICY_PATH", str(p))
        ag = IndicatorAgent()
        assert ag.policy["type2_rules"] == {}
        assert ag.policy["score"] == 3                    # nothing else touched
