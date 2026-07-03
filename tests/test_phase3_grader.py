"""Phase 3 verification: the auto-labeling grader + manual-override precedence.

- auto: a due prediction is graded from realized OHLCV; each agent is rewarded
  against its OWN stored payload; outcome/rewards persisted; row marked graded.
- manual from pending: human label applied directly, grader leaves it alone.
- manual correcting auto: correction = manual - auto, so the NET policy effect
  equals the human verdict.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ----- test doubles ------------------------------------------------------- #
class FakeAgent:
    def __init__(self):
        self.calls = []

    def apply_reward(self, *args):
        self.calls.append(args)        # news/research: (feats, idx, reward); indicator: (blend, reward)


class FakeDM:
    def __init__(self):
        self.news = FakeAgent()
        self.research = FakeAgent()
        self.indicator = FakeAgent()
        self.brain_calls = []

    def apply_brain_feedback(self, agent_results, label, news_reward):
        self.brain_calls.append((agent_results, label, news_reward))


def make_decision(news_action="BUY", research_action="buy", indicator_action="buy"):
    return {
        "chartName": "BTCUSDT", "timeframe": "4h",
        "agents": {
            "news": {"action": news_action, "confidence": 0.9,
                     "raw": {"action": news_action, "rl": {"features": [0.1, 0.2, 0.0, 1.0, 0.0], "action_idx": 2}}},
            "research": {"action": research_action, "confidence": 0.8,
                         "raw": {"action": research_action, "rl": {"feats": [0.1] * 10, "action_idx": 2}}},
            "indicator": {"action": indicator_action, "confidence": 0.7,
                          "raw": {"action": indicator_action,
                                  "details": {"blend": {"type1_share": 0.6, "type2_share": 0.4, "fired_direct": "nwe"}}}},
        },
        "final": {"action": "buy", "confidence": 0.85, "score": 0.5},
        "policy": {"weights": {"indicator": 0.6, "research": 0.3, "news": 0.1}},
    }


def make_fetcher(entry_close=100.0, horizon_close=105.0, tf="4h", k=2):
    """Fetcher whose k-th candle after a chosen close has ``horizon_close``."""
    n = 20
    ts = pd.date_range("2024-01-01", periods=n, freq=tf)
    closes = [100.0] * n
    close_idx = 5
    closes[close_idx] = entry_close
    closes[close_idx + k] = horizon_close
    df = pd.DataFrame({"timestamp": ts, "open": closes, "high": closes,
                       "low": closes, "close": closes, "volume": [1.0] * n})
    # correctness v3 convention: candle_close_ts = entry candle CLOSE epoch
    close_ts = int((ts[close_idx + 1] - pd.Timestamp("1970-01-01")) // pd.Timedelta("1s"))

    class _F:
        def get_ohlcv(self, pair, tf_, limit=500):
            return df.copy()

    return _F(), close_ts


# ----- tests -------------------------------------------------------------- #
def test_auto_grade(tmp_path):
    from persistence import Store
    from grader import Grader
    store = Store(str(tmp_path / "g.db"))
    fetcher, close_ts = make_fetcher(100.0, 105.0, "4h", 2)      # +5% -> realized buy
    dm = FakeDM()
    g = Grader(dm, data_fetcher=fetcher, store=store)

    dec = make_decision(news_action="BUY", research_action="buy", indicator_action="sell")
    pid = store.record_prediction(dec, candle_close_ts=close_ts, entry_price=100.0,
                                  horizon_k=2, grade_due_ts=1.0)

    out = g.grade_once(now_ts=close_ts + 10 ** 6)
    assert len(out) == 1
    r = out[0]
    assert r["realized_label"] == "buy"
    assert abs(r["forward_return"] - 0.05) < 1e-9
    # news BUY==buy -> +1 ; research buy -> +1 ; indicator sell != buy -> -4
    assert r["rewards"] == {"news": 1.0, "research": 1.0, "indicator": -4.0}

    assert store.get_outcome(pid)["realized_label"] == "buy"
    p = store.get_prediction(pid)
    assert p["label_source"] == "auto" and p["graded"] == 1
    # each agent trained from its stored payload; brain learned once
    assert dm.news.calls and dm.research.calls and dm.indicator.calls and len(dm.brain_calls) == 1
    # no longer due
    assert g.grade_once(now_ts=close_ts + 10 ** 6) == []


def test_manual_from_pending(tmp_path):
    from persistence import Store
    from grader import Grader
    store = Store(str(tmp_path / "g2.db"))
    dm = FakeDM()
    g = Grader(dm, data_fetcher=None, store=store)

    pid = store.record_prediction(make_decision(), candle_close_ts=100.0, entry_price=100.0,
                                  horizon_k=2, grade_due_ts=1.0)
    res = g.apply_manual_feedback(pid, "sell", news_reward=-4.0)
    assert res["status"] == "manual"

    p = store.get_prediction(pid)
    assert p["label_source"] == "manual" and p["graded"] == 1
    rw = store.rewards_for(pid)
    assert rw and all(r["source"] == "manual" for r in rw)
    assert any(r["agent"] == "news" and r["reward"] == -4.0 for r in rw)   # explicit numeric honored
    # grader never re-touches a manual row
    assert g.grade_once(now_ts=10 ** 9) == []


def test_manual_corrects_auto(tmp_path):
    from persistence import Store
    from grader import Grader
    store = Store(str(tmp_path / "g3.db"))
    fetcher, close_ts = make_fetcher(100.0, 105.0, "4h", 2)      # auto -> buy, all correct (+1)
    dm = FakeDM()
    g = Grader(dm, data_fetcher=fetcher, store=store)

    dec = make_decision(news_action="BUY", research_action="buy", indicator_action="buy")
    pid = store.record_prediction(dec, candle_close_ts=close_ts, entry_price=100.0,
                                  horizon_k=2, grade_due_ts=1.0)
    g.grade_once(now_ts=close_ts + 10 ** 6)
    auto = {r["agent"]: r["reward"] for r in store.rewards_for(pid) if r["source"] == "auto"}
    assert auto["news"] == 1.0 and auto["research"] == 1.0

    # human disagrees: actually SELL, with explicit news reward -4.
    res = g.apply_manual_feedback(pid, "sell", news_reward=-4.0)
    assert res["status"] == "corrected"
    assert store.get_prediction(pid)["label_source"] == "manual"

    corr = {r["agent"]: r["reward"] for r in store.rewards_for(pid) if r["source"] == "correction"}
    # news: manual -4 - auto +1 = -5 ; research: manual -4 (buy!=sell) - auto +1 = -5
    assert corr["news"] == -5.0 and corr["research"] == -5.0
    # net reward delivered to the news agent (auto + correction) equals the manual verdict
    news_rewards = [c[2] for c in dm.news.calls]      # (feats, idx, reward)
    assert abs(sum(news_rewards) - (-4.0)) < 1e-9
