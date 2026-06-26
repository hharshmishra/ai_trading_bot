"""Phase 2b verification: durable SQLite state + the race fix end-to-end.

1. 48 threads record predictions concurrently, each with a UNIQUE RL payload.
   Every row must round-trip intact, keyed by its own id — exactly what the old
   shared-singleton state could not do.
2. The grader pattern (due query -> per-prediction reward/outcome -> mark graded)
   operates on each prediction's OWN stored data.
3. A real agent's apply_reward, fed the STORED payload, trains on that prediction.
"""
from __future__ import annotations

import copy
import sys
import threading
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def make_decision(i: int) -> dict:
    """A synthetic brain.decide() output with an i-stamped, unique RL payload."""
    news_feats = [float(i), 0.0, 0.0, 0.0, 0.0]       # 5-dim; [0] is the signature
    research_feats = [float(i)] + [0.0] * 9            # 10-dim; [0] is the signature
    blend = {"type1_share": 0.5, "type2_share": 0.5, "fired_direct": f"sig{i}"}
    return {
        "chartName": f"P{i}USDT", "timeframe": "4h",
        "agents": {
            "news": {"action": "buy", "confidence": 0.9,
                     "raw": {"action": "BUY", "rl": {"features": news_feats, "action_idx": 2}}},
            "research": {"action": "buy", "confidence": 0.8,
                         "raw": {"action": "buy", "rl": {"feats": research_feats, "action_idx": 2}}},
            "indicator": {"action": "buy", "confidence": 0.7,
                          "raw": {"action": "buy", "details": {"blend": blend}}},
        },
        "final": {"action": "buy", "confidence": 0.85, "score": 0.5},
        "policy": {"weights": {"indicator": 0.6, "research": 0.3, "news": 0.1}},
    }


def test_concurrent_record_no_contamination(tmp_path):
    from persistence import Store
    store = Store(str(tmp_path / "t.db"))
    N = 48
    ids: dict[int, str] = {}
    errors: list = []

    def worker(i):
        try:
            ids[i] = store.record_prediction(
                make_decision(i), cycle_id="c1", candle_close_ts=1000.0 + i,
                entry_price=100.0 + i, horizon_k=2, grade_due_ts=1.0)
        except Exception as e:  # pragma: no cover
            errors.append((i, repr(e)))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(N)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, errors
    assert len(ids) == N

    # Every prediction's stored RL payload matches ITS OWN i — no cross-contamination.
    for i, pid in ids.items():
        p = store.get_prediction(pid)
        assert p["pair"] == f"P{i}USDT"
        assert p["news_feats"][0] == float(i)
        assert p["research_feats"][0] == float(i)
        assert p["indicator_blend"]["fired_direct"] == f"sig{i}"
        assert p["entry_price"] == 100.0 + i
        assert p["label_source"] == "pending" and p["graded"] == 0
    store.close()


def test_grader_pattern_per_prediction(tmp_path):
    from persistence import Store
    store = Store(str(tmp_path / "t2.db"))
    for i in range(10):
        store.record_prediction(make_decision(i), grade_due_ts=1.0, horizon_k=2)

    due = store.get_due_predictions(now_ts=2.0)
    assert len(due) == 10

    for p in due:
        store.record_outcome(p["id"], realized_return=0.05, realized_label="buy",
                             threshold=0.01, horizon_k=p["horizon_k"])
        store.record_reward(p["id"], "news", p["news_action"], 1.0, source="auto")
        store.mark_graded(p["id"], "auto")

    assert store.get_due_predictions(now_ts=2.0) == []     # nothing left to grade
    for p in due:
        rw = store.rewards_for(p["id"])
        assert len(rw) == 1 and rw[0]["agent"] == "news" and rw[0]["source"] == "auto"
        assert store.get_prediction(p["id"])["label_source"] == "auto"
    store.close()


def test_stored_payload_drives_real_agent_update(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "logs").mkdir()
    from persistence import Store
    from agents.news_agent import NewsAgent

    store = Store(str(tmp_path / "logs" / "t3.db"))
    pid = store.record_prediction(make_decision(7), grade_due_ts=1.0, horizon_k=2)
    p = store.get_prediction(pid)

    ag = NewsAgent()
    W0 = copy.deepcopy(ag._rl.policy.weights)
    ag.apply_reward(p["news_feats"], p["news_action_idx"], 1.0)   # train from STORED data
    assert ag._rl.policy.weights != W0                            # it trained on this prediction
    store.close()


def test_session_supersede_and_gc(tmp_path):
    from persistence import Store
    store = Store(str(tmp_path / "t4.db"))
    s1 = store.create_session(pair="BTCUSDT", tf="4h", created_ts=100.0)
    assert store.get_active_session("BTCUSDT", "4h")["id"] == s1

    s2 = store.create_session(pair="BTCUSDT", tf="4h", created_ts=200.0)
    prev = store.supersede_active("BTCUSDT", "4h", s2)
    assert prev == s1
    assert store.get_session(s1)["active"] == 0
    assert store.get_session(s1)["superseded_by"] == s2

    # GC closes sessions older than cutoff
    closed = store.gc_sessions(cutoff_ts=150.0)   # s2 created at 200 stays; nothing < 150 active
    assert closed == []
    store.gc_sessions(cutoff_ts=300.0)
    assert store.get_active_session("BTCUSDT", "4h") is None
    store.close()
