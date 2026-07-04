"""v3.2 learning-criteria fixes: brain uses the active reward map (no news
double-count), indicator steps scale with reward magnitude, one-tap VERDICT
trains everything including FLAT."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config

T0 = 1_800_000_000.0


# --------------------------- brain criteria (C1/C2) ------------------------ #
class TestBrainRewardMap:
    def _dm(self):
        from brain.decision_maker import DecisionMaker
        return DecisionMaker(prefer_csv=False)

    def test_flat_outcome_uses_v2_timeout_not_full_wrong(self, monkeypatch):
        monkeypatch.setattr(config, "TB_GRADING_ENABLED", True)
        dm = self._dm()
        before = dict(dm.policy["scores"])
        results = {"indicator": {"action": "buy", "confidence": 1.0}}
        dm.apply_brain_feedback(results, "skip")
        delta = dm.policy["scores"]["indicator"] - before["indicator"]
        # v2 map: directional-vs-flat = REWARD_TIMEOUT_FLAT (-1.5), not -4
        assert delta == pytest.approx(0.05 * config.REWARD_TIMEOUT_FLAT * 1.0)

    def test_legacy_map_when_tb_off(self, monkeypatch):
        monkeypatch.setattr(config, "TB_GRADING_ENABLED", False)
        dm = self._dm()
        before = dict(dm.policy["scores"])
        dm.apply_brain_feedback({"indicator": {"action": "buy", "confidence": 1.0}}, "skip")
        delta = dm.policy["scores"]["indicator"] - before["indicator"]
        assert delta == pytest.approx(0.05 * config.REWARD_WRONG)     # v1: flat wrong

    def test_news_scored_once_no_double_count(self, monkeypatch):
        monkeypatch.setattr(config, "TB_GRADING_ENABLED", True)
        dm = self._dm()
        before = dict(dm.policy["scores"])
        dm.apply_brain_feedback({"news": {"action": "buy", "confidence": 0.8}}, "buy")
        delta = dm.policy["scores"]["news"] - before["news"]
        # scored exactly once (the old code added the news reward a second time)
        assert delta == pytest.approx(0.05 * config.REWARD_CORRECT * 0.8)


# ------------------------ indicator criteria (C3) -------------------------- #
class TestIndicatorMagnitude:
    def _fresh_agent(self):
        from agents.indicator_agent import IndicatorAgent
        return IndicatorAgent()

    def _blend(self, fired=None):
        b = {"type1_share": 0.5, "type2_share": 0.5}
        if fired:
            b["fired_direct"] = fired
        return b

    def test_wrong_direction_moves_four_times_missed_move(self):
        ag1 = self._fresh_agent()
        w0 = ag1.policy["weights"]["type1"]
        ag1.apply_reward(self._blend(), -4.0)
        d_wrong = abs(ag1.policy["weights"]["type1"] - w0)

        # fresh policy for the second measurement (conftest isolates the path,
        # but the file persists within one test) — reset by rewriting
        from agents import indicator_agent as ia
        import os
        os.remove(ia.POLICY_PATH)
        ag2 = self._fresh_agent()
        w0b = ag2.policy["weights"]["type1"]
        ag2.apply_reward(self._blend(), -1.0)
        d_miss = abs(ag2.policy["weights"]["type1"] - w0b)

        assert d_wrong > d_miss                                  # severity reaches policy
        assert d_wrong / d_miss == pytest.approx(4.0, rel=0.25)  # ~|r| ratio (post-norm)

    def test_direct_signal_weight_scales_and_clips(self):
        ag = self._fresh_agent()
        ag.apply_reward(self._blend(fired="nwe"), -4.0)
        w_after_big = ag.policy["direct_signals"]["nwe"]["weight"]
        assert w_after_big == pytest.approx(0.7 - 0.07)          # wrong == historical step
        for _ in range(20):
            ag.apply_reward(self._blend(fired="nwe"), -4.0)
        assert ag.policy["direct_signals"]["nwe"]["weight"] >= 0.1  # clip holds


# ------------------------- one-tap VERDICT (C4) ---------------------------- #
class FakeAgent:
    def __init__(self):
        self.calls = []

    def apply_reward(self, *a):
        self.calls.append(a)


class FakeDM:
    def __init__(self):
        self.news = FakeAgent(); self.research = FakeAgent(); self.indicator = FakeAgent()
        self.brain = []

    def apply_brain_feedback(self, *a):
        self.brain.append(a)


def _decision():
    return {
        "chartName": "BTCUSDT", "timeframe": "4h",
        "agents": {
            "news": {"action": "buy", "confidence": 0.7,
                     "raw": {"action": "buy", "rl": {"features": [0.1] * 5, "action_idx": 2}}},
            "research": {"action": "skip", "confidence": 0.6,
                         "raw": {"action": "skip", "rl": {"feats": [0.1] * 10, "action_idx": 1}}},
            "indicator": {"action": "buy", "confidence": 0.8,
                          "raw": {"action": "buy",
                                  "details": {"blend": {"type1_share": 0.6, "fired_direct": "nwe"}}}},
        },
        "final": {"action": "buy", "confidence": 0.8, "score": 0.5},
        "policy": {"weights": {"indicator": 0.6}},
    }


class CQ:
    def __init__(self, data):
        self.data = data
        self.answers = []

    async def answer(self, text="", show_alert=False):
        self.answers.append(text)


def _harness(tmp_path):
    from grader import Grader
    from persistence import Store
    from telegram_app import Broadcaster
    store = Store(str(tmp_path / "v.db"))
    dm = FakeDM()
    grader = Grader(dm, data_fetcher=None, store=store)

    class _B:                       # minimal fake bot for Broadcaster.strip
        async def edit_message_reply_markup(self, **kw):
            pass
    bc = Broadcaster(_B(), store, 111, 222)
    ctx = SimpleNamespace(application=SimpleNamespace(
        bot_data={"store": store, "grader": grader, "broadcaster": bc}))
    return store, dm, grader, ctx


def test_one_tap_flat_trains_all_agents_with_map(tmp_path, monkeypatch):
    """C4: FLAT is a real verdict — one tap grades every agent + brain."""
    monkeypatch.setattr(config, "TB_GRADING_ENABLED", True)
    from telegram_app import handle_callback
    store, dm, grader, ctx = _harness(tmp_path)
    # mirror the production order exactly: session created first (broadcast),
    # prediction recorded WITH session_id, then the reverse link — so the
    # grader (the single writer of true_outcome) can set it on p["session_id"].
    sid = store.create_session(pair="BTCUSDT", tf="4h", dev_chat_id=222, dev_msg_id=9)
    pid = store.record_prediction(_decision(), candle_close_ts=100.0, entry_price=100.0,
                                  horizon_k=2, grade_due_ts=1.0, session_id=sid)
    store.link_session_prediction(sid, pid)

    cq = CQ(f"{sid}|VERDICT|skip")
    asyncio.run(handle_callback(SimpleNamespace(callback_query=cq), ctx))

    p = store.get_prediction(pid)
    assert p["label_source"] == "manual"
    rewards = {r["agent"]: r["reward"] for r in store.rewards_for(pid)}
    assert rewards["news"] == config.REWARD_TIMEOUT_FLAT          # buy vs flat
    assert rewards["research"] == config.REWARD_CORRECT           # skip vs flat
    assert rewards["indicator"] == config.REWARD_TIMEOUT_FLAT
    assert dm.news.calls and dm.research.calls and dm.indicator.calls
    assert dm.brain                                               # brain trained too
    assert store.get_session(sid)["active"] == 0
    assert store.get_session(sid)["true_outcome"] == "skip"
    assert "FLAT" in cq.answers[-1]


def test_one_tap_verdict_on_null_pid_keeps_session(tmp_path):
    from telegram_app import handle_callback
    store, dm, grader, ctx = _harness(tmp_path)
    sid = store.create_session(pair="BTCUSDT", tf="4h", dev_chat_id=222, dev_msg_id=9)
    cq = CQ(f"{sid}|VERDICT|buy")
    asyncio.run(handle_callback(SimpleNamespace(callback_query=cq), ctx))
    assert "try again" in cq.answers[-1]
    assert store.get_session(sid)["active"] == 1


def test_one_tap_on_auto_graded_row_corrects(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "TB_GRADING_ENABLED", True)
    from telegram_app import handle_callback
    store, dm, grader, ctx = _harness(tmp_path)
    pid = store.record_prediction(_decision(), candle_close_ts=100.0, entry_price=100.0,
                                  horizon_k=2, grade_due_ts=1.0)
    sid = store.create_session(pair="BTCUSDT", tf="4h", prediction_id=pid,
                               dev_chat_id=222, dev_msg_id=9)
    # simulate a prior auto grade vs 'buy'
    assert store.claim_grading(pid, "auto")
    grader._apply_rewards(store.get_prediction(pid), "buy", source="auto")

    cq = CQ(f"{sid}|VERDICT|skip")                                # human: it was flat
    asyncio.run(handle_callback(SimpleNamespace(callback_query=cq), ctx))
    p = store.get_prediction(pid)
    assert p["label_source"] == "manual"
    assert "corrected" in cq.answers[-1]
