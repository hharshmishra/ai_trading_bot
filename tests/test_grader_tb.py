"""Phase 3: triple-barrier grading v2 — reward map, legacy fallback, manual precedence."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config


class FakeAgent:
    def __init__(self):
        self.calls = []

    def apply_reward(self, *args):
        self.calls.append(args)


class FakeDM:
    def __init__(self):
        self.news = FakeAgent()
        self.research = FakeAgent()
        self.indicator = FakeAgent()
        self.brain_calls = []

    def apply_brain_feedback(self, agent_results, label, news_reward):
        self.brain_calls.append((agent_results, label, news_reward))


def make_decision(news="buy", research="buy", indicator="buy", final="buy",
                  regime="ranging", atr=2.0):
    return {
        "chartName": "BTCUSDT", "timeframe": "4h",
        "agents": {
            "news": {"action": news, "confidence": 0.9,
                     "raw": {"action": news, "rl": {"features": [0.1] * 5, "action_idx": 2}}},
            "research": {"action": research, "confidence": 0.8,
                         "raw": {"action": research, "rl": {"feats": [0.1] * 10, "action_idx": 2}}},
            "indicator": {"action": indicator, "confidence": 0.7,
                          "raw": {"action": indicator,
                                  "details": {"blend": {"type1_share": 0.6, "fired_direct": "nwe"},
                                              "regime": regime,
                                              "regime_feats": {"atr": atr, "vol_ok": True}}}},
        },
        "final": {"action": final, "confidence": 0.85, "score": 0.5},
        "policy": {"weights": {"indicator": 0.6}},
    }


def make_path_fetcher(entry_close, path_rows, tf="4h"):
    """Fetcher: candle[5] closes at entry; the following candles follow path_rows
    [(high, low, close), ...]."""
    n = 6 + len(path_rows)
    ts = pd.date_range("2024-01-01", periods=n, freq=tf)
    highs, lows, closes = [], [], []
    for i in range(6):
        c = entry_close if i == 5 else 100.0
        highs.append(c); lows.append(c); closes.append(c)
    for h, l, c in path_rows:
        highs.append(h); lows.append(l); closes.append(c)
    df = pd.DataFrame({"timestamp": ts, "open": closes, "high": highs,
                       "low": lows, "close": closes, "volume": [1.0] * n})
    # correctness v3 convention: candle_close_ts = entry candle CLOSE epoch
    # (= open of the next candle); grader filters path rows with ts >= close_ts
    close_ts = int((ts[6] - pd.Timestamp("1970-01-01")) // pd.Timedelta("1s"))

    class _F:
        def get_ohlcv(self, pair, tf_, limit=500):
            return df.copy()

    return _F(), close_ts


def _record(store, dec, close_ts, entry=100.0, tp=103.0, sl=98.0, k=2):
    return store.record_prediction(dec, candle_close_ts=close_ts, entry_price=entry,
                                   horizon_k=k, grade_due_ts=1.0,
                                   tp_price=tp, sl_price=sl)


@pytest.fixture
def tb_on(monkeypatch):
    monkeypatch.setattr(config, "TB_GRADING_ENABLED", True)


def _grade(tmp_path, dec, fetcher, close_ts, record_kwargs=None):
    from persistence import Store
    from grader import Grader
    store = Store(str(tmp_path / "g.db"))
    dm = FakeDM()
    g = Grader(dm, data_fetcher=fetcher, store=store)
    pid = store.record_prediction(dec, candle_close_ts=close_ts, entry_price=100.0,
                                  horizon_k=2, grade_due_ts=1.0, **(record_kwargs or {}))
    out = g.grade_once(now_ts=close_ts + 10 ** 6)
    return store, dm, pid, out


class TestTbGrading:
    def test_tp_first_buy_rewards_plus_one(self, tmp_path, tb_on):
        # candle 1 hits TP 103 (high 103.5) without touching SL 98
        fetcher, close_ts = make_path_fetcher(100.0, [(103.5, 99.5, 103.2), (104, 102, 103.8)])
        store, dm, pid, out = _grade(
            tmp_path, make_decision(), fetcher, close_ts,
            {"tp_price": 103.0, "sl_price": 98.0})
        assert out[0]["label_tb"] == "tp"
        assert out[0]["realized_label"] == "buy"
        assert out[0]["rewards"] == {"news": 1.0, "research": 1.0, "indicator": 1.0}
        o = store.get_outcome(pid)
        assert o["label_tb"] == "tp" and o["barrier_hit_idx"] == 1

    def test_sl_first_buy_rewards_minus_four(self, tmp_path, tb_on):
        # candle 1 hits SL 98 first; fixed-horizon close ends ABOVE entry —
        # path-awareness is what flips this to a loss
        fetcher, close_ts = make_path_fetcher(100.0, [(100.5, 97.5, 99.0), (104, 99, 103.5)])
        store, dm, pid, out = _grade(
            tmp_path, make_decision(), fetcher, close_ts,
            {"tp_price": 103.0, "sl_price": 98.0})
        assert out[0]["label_tb"] == "sl"
        assert out[0]["realized_label"] == "sell"
        assert out[0]["rewards"]["indicator"] == config.REWARD_WRONG
        # fixed-horizon return still recorded for comparability
        assert store.get_outcome(pid)["realized_return"] == pytest.approx(0.035)

    def test_timeout_flat_directional_gets_minus_1_5(self, tmp_path, tb_on):
        # neither barrier, k-th close ~flat (|fr| < 1% threshold for 4h)
        fetcher, close_ts = make_path_fetcher(100.0, [(101, 99.2, 100.3), (101, 99.5, 100.2)])
        store, dm, pid, out = _grade(
            tmp_path, make_decision(), fetcher, close_ts,
            {"tp_price": 103.0, "sl_price": 98.0})
        assert out[0]["label_tb"] == "timeout"
        assert out[0]["realized_label"] == "skip"
        assert out[0]["rewards"]["indicator"] == config.REWARD_TIMEOUT_FLAT

    def test_skip_prediction_while_market_moved_gets_minus_1(self, tmp_path, tb_on):
        # all agents skipped; no barriers recorded (final=skip); market rallied 5%
        dec = make_decision(news="skip", research="skip", indicator="skip", final="skip")
        fetcher, close_ts = make_path_fetcher(100.0, [(103, 100, 102.5), (105.5, 102, 105.0)])
        store, dm, pid, out = _grade(tmp_path, dec, fetcher, close_ts)
        assert out[0]["label_tb"] is None
        assert out[0]["realized_label"] == "buy"
        assert out[0]["rewards"]["indicator"] == config.REWARD_MISSED_MOVE

    def test_ambiguous_candle_is_sl(self, tmp_path, tb_on):
        # candle 1 spans both barriers -> pessimistic SL
        fetcher, close_ts = make_path_fetcher(100.0, [(103.5, 97.5, 100.0), (101, 99, 100.5)])
        store, dm, pid, out = _grade(
            tmp_path, make_decision(), fetcher, close_ts,
            {"tp_price": 103.0, "sl_price": 98.0})
        assert out[0]["label_tb"] == "sl"
        assert out[0]["realized_label"] == "sell"

    def test_legacy_row_without_barriers_uses_fixed_horizon(self, tmp_path, tb_on):
        # pre-migration row: NULL tp/sl -> fixed-horizon label even with flag on
        fetcher, close_ts = make_path_fetcher(100.0, [(105, 100, 104.0), (106, 103, 105.0)])
        store, dm, pid, out = _grade(tmp_path, make_decision(), fetcher, close_ts)
        assert out[0]["label_tb"] is None
        assert out[0]["realized_label"] == "buy"     # +5% fixed horizon
        assert out[0]["rewards"]["indicator"] == 1.0

    def test_flag_off_records_tb_but_rewards_v1(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "TB_GRADING_ENABLED", False)
        # SL-first path, but flag off -> rewards from fixed-horizon (+3.5% = buy)
        fetcher, close_ts = make_path_fetcher(100.0, [(100.5, 97.5, 99.0), (104, 99, 103.5)])
        store, dm, pid, out = _grade(
            tmp_path, make_decision(), fetcher, close_ts,
            {"tp_price": 103.0, "sl_price": 98.0})
        assert out[0]["realized_label"] == "buy"          # v1 label drives rewards
        assert out[0]["rewards"]["indicator"] == 1.0
        assert store.get_outcome(pid)["label_tb"] == "sl"  # shadow evidence recorded

    def test_manual_override_still_wins_after_tb_auto(self, tmp_path, tb_on):
        from persistence import Store
        from grader import Grader
        fetcher, close_ts = make_path_fetcher(100.0, [(103.5, 99.5, 103.2), (104, 102, 103.8)])
        store = Store(str(tmp_path / "g.db"))
        dm = FakeDM()
        g = Grader(dm, data_fetcher=fetcher, store=store)
        pid = store.record_prediction(make_decision(), candle_close_ts=close_ts,
                                      entry_price=100.0, horizon_k=2, grade_due_ts=1.0,
                                      tp_price=103.0, sl_price=98.0)
        g.grade_once(now_ts=close_ts + 10 ** 6)          # auto: tp -> +1 each
        res = g.apply_manual_feedback(pid, "sell")        # human disagrees
        assert res["status"] == "corrected"
        # net indicator effect must equal the human verdict (-4): correction = -4 - (+1)
        assert res["corrections"]["indicator"] == pytest.approx(-5.0)
        assert store.get_prediction(pid)["label_source"] == "manual"


class TestRewardMapV2:
    def test_map(self):
        from grader import reward_for_v2
        assert reward_for_v2("buy", "buy") == config.REWARD_CORRECT
        assert reward_for_v2("sell", "buy") == config.REWARD_WRONG
        assert reward_for_v2("buy", "skip") == config.REWARD_TIMEOUT_FLAT
        assert reward_for_v2("skip", "buy") == config.REWARD_MISSED_MOVE
        assert reward_for_v2("skip", "skip") == config.REWARD_CORRECT
