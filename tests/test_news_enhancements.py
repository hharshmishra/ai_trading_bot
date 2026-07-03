"""Phase C: event-typed extraction, RL 5->10 migration, headline formatting."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config


class TestEventFeatures:
    def test_no_events_all_zero(self):
        from agents.news_agent import _event_features
        assert _event_features([]) == [0.0, 0.0, 0.0, 0.0, 0.0]

    def test_event_mapping(self):
        from agents.news_agent import NewsEvent, _event_features
        events = [
            NewsEvent(type="hack", direction="Bearish", surprise=0.9, source_tier=1),
            NewsEvent(type="etf_flow", direction="Bullish", surprise=0.8, source_tier=2),
            NewsEvent(type="unlock", direction="Neutral", surprise=0.5, source_tier=3),
        ]
        bull, bear, hack_reg, etf, unlock = _event_features(events)
        assert bear == pytest.approx(0.9 * 1.0)     # hack, tier-1
        assert bull == pytest.approx(0.8 * 0.7)     # etf, tier-2
        assert hack_reg == 1.0 and unlock == 1.0
        assert etf == pytest.approx(0.8 * 0.7)      # signed bullish

    def test_features_from_jsons_is_10_dim(self):
        from agents.news_agent import (OverallScanJSON, PairScanJSON,
                                       features_from_jsons)
        o = OverallScanJSON(has_panic=False, sentiment="Neutral", confidence=0.5)
        p = PairScanJSON(pair="BTCUSDT", sentiment="Bullish", confidence=0.6)
        feats = features_from_jsons(o, p)
        assert len(feats) == 10
        assert feats[5:] == [0.0] * 5   # no events


class TestRlMigration:
    def _write_5dim_policy(self, path):
        weights = [[0.1, 0.2, 0.3, 0.4, 0.5],
                   [-0.1, -0.2, -0.3, -0.4, -0.5],
                   [0.05, 0.0, -0.05, 0.1, 0.0]]
        path.write_text(json.dumps({"weights": weights, "epsilon": 0.1}))
        return weights

    def test_zero_pad_preserves_logits_exactly(self, tmp_path, monkeypatch):
        import agents.news_agent as na
        pol_path = tmp_path / "news_policy.json"
        old_weights = self._write_5dim_policy(pol_path)
        monkeypatch.setattr(na, "POLICY_PATH", str(pol_path))

        rl = na.NewsRL()
        assert len(rl.policy.weights[0]) == 10
        feats5 = [0.3, -0.2, 1.0, 1.0, -1.0]
        old_logits = [sum(w * f for w, f in zip(row, feats5)) for row in old_weights]
        new_logits = rl._logits(rl._pad(feats5))
        assert new_logits == pytest.approx(old_logits)

    def test_migration_writes_backup_and_is_idempotent(self, tmp_path, monkeypatch):
        import agents.news_agent as na
        pol_path = tmp_path / "news_policy.json"
        self._write_5dim_policy(pol_path)
        monkeypatch.setattr(na, "POLICY_PATH", str(pol_path))

        na.NewsRL()
        assert (tmp_path / "news_policy.json.bak-5dim").exists()
        saved = json.loads(pol_path.read_text())
        assert saved["n_features"] == 10 and len(saved["weights"][0]) == 10

        rl2 = na.NewsRL()   # second load: already migrated, no change
        assert len(rl2.policy.weights[0]) == 10

    def test_update_pads_stored_5dim_rows(self, tmp_path, monkeypatch):
        import agents.news_agent as na
        pol_path = tmp_path / "news_policy.json"
        self._write_5dim_policy(pol_path)
        monkeypatch.setattr(na, "POLICY_PATH", str(pol_path))
        rl = na.NewsRL()
        rl.update([0.1, 0.2, 0.3, 0.4, 0.5], 2, 1.0)   # old stored row: no IndexError
        # event-feature weight slots untouched by a 5-dim replay
        assert all(rl.policy.weights[a][j] == 0.0 for a in range(3) for j in range(5, 10))


class TestAgentWiring:
    def test_agent_bandit_width_follows_n_features(self):
        """Unit tests construct NewsRL() directly; the AGENT ctor is the live
        wiring. It once pinned n_features=5 while features_from_jsons returns
        10 dims — _pad silently truncated event features and the 5->10 policy
        migration never ran in production."""
        import agents.news_agent as na
        ag = na.NewsAgent()
        assert ag._rl.n_features == na.N_FEATURES


class TestEventsPromptFlag:
    def test_flag_off_prompt_unchanged(self, monkeypatch):
        import agents.news_agent as na
        monkeypatch.setattr(config, "NEWS_EVENTS_ENABLED", False)
        prompts = []
        monkeypatch.setattr(na, "_chat_json", lambda p: (prompts.append(p) or {
            "has_panic": False, "sentiment": "Neutral", "confidence": 0.5,
            "top_headlines": []}))
        na.NewsAgent().scan_overall(headlines=["x"])
        assert "events" not in prompts[-1].split("Recent market headlines")[0]

    def test_flag_on_requests_events(self, monkeypatch):
        import agents.news_agent as na
        monkeypatch.setattr(config, "NEWS_EVENTS_ENABLED", True)
        prompts = []
        monkeypatch.setattr(na, "_chat_json", lambda p: (prompts.append(p) or {
            "has_panic": False, "sentiment": "Neutral", "confidence": 0.5,
            "top_headlines": [], "events": [
                {"type": "hack", "direction": "Bearish", "surprise": 0.9, "source_tier": 1}]}))
        out = na.NewsAgent().scan_overall(headlines=["x"])
        assert '"events"' in prompts[-1]
        assert out.events[0].type == "hack"

    def test_old_llm_output_without_events_still_valid(self):
        from agents.news_agent import OverallScanJSON
        o = OverallScanJSON.model_validate(
            {"has_panic": False, "sentiment": "Neutral", "confidence": 0.5,
             "top_headlines": []})
        assert o.events == []


class TestHeadlineFormatting:
    def test_format_headline(self):
        import time
        from ingestion import format_headline
        now = time.time()
        row = {"title": "T", "source": "coindesk", "published_ts": now - 3 * 3600}
        assert format_headline(row, now=now) == "[3h ago] [tier-1] T"
        row2 = {"title": "U", "source": "unknownblog", "published_ts": None}
        assert format_headline(row2, now=now) == "[undated] [tier-2] U"
        row3 = {"title": "V", "source": "newsbtc", "published_ts": now - 60 * 3600}
        assert format_headline(row3, now=now) == "[2d ago] [tier-3] V"
