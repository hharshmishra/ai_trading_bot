"""Correctness v3 (Phase A): closed-candle discipline, close-epoch convention,
grader boundary, event-mode NWE."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config


def _frame(n, freq, end_at_now=False):
    if end_at_now:
        # last candle opened "now" minus a fraction of its duration -> partial
        step = pd.Timedelta(freq)
        end = pd.Timestamp.utcnow().tz_localize(None).floor(freq)
        idx = pd.date_range(end=end, periods=n, freq=freq)
    else:
        idx = pd.date_range("2024-01-01", periods=n, freq=freq)
    close = np.linspace(100, 110, n)
    return pd.DataFrame({"timestamp": idx, "open": close, "high": close + 1,
                         "low": close - 1, "close": close, "volume": np.ones(n)})


class TestDropPartial:
    def test_drops_in_progress_candle(self):
        from utils.data_fetcher import _drop_partial_candle
        df = _frame(10, "1h", end_at_now=True)   # last row opened <1h ago
        out = _drop_partial_candle(df, "1h")
        assert len(out) == 9
        assert out["timestamp"].iloc[-1] == df["timestamp"].iloc[-2]

    def test_keeps_historical_frames(self):
        from utils.data_fetcher import _drop_partial_candle
        df = _frame(10, "1h")                     # 2024 candles, all closed
        out = _drop_partial_candle(df, "1h")
        assert len(out) == 10

    def test_unknown_timeframe_untouched(self):
        from utils.data_fetcher import _drop_partial_candle
        df = _frame(5, "1h", end_at_now=True)
        assert len(_drop_partial_candle(df, "3h")) == 5


class TestCloseEpochConvention:
    def test_entry_from_df_returns_close_epoch(self):
        from cycle import _entry_from_df
        df = _frame(6, "4h")
        entry, close_ts = _entry_from_df(df, "4h")
        last_open = int((df["timestamp"].iloc[-1] - pd.Timestamp("1970-01-01"))
                        // pd.Timedelta("1s"))
        assert close_ts == last_open + 14400
        assert entry == pytest.approx(float(df["close"].iloc[-1]))

    def test_grade_due_walkthrough(self):
        # entry candle opens 03:00, closes 04:00 (close_ts). k=3 on 1h ->
        # due at 07:00, when candles opening 04:00/05:00/06:00 are all closed.
        from signals import TF_SECONDS
        open_epoch = int(pd.Timestamp("2024-01-01 03:00").timestamp())
        close_ts = open_epoch + TF_SECONDS["1h"]
        due = close_ts + 3 * TF_SECONDS["1h"]
        assert pd.Timestamp(due, unit="s") == pd.Timestamp("2024-01-01 07:00")


class TestGraderBoundary:
    def test_first_path_candle_opens_at_close_ts(self):
        """Candle opening exactly AT candle_close_ts is the first path candle."""
        from grader import Grader

        df = _frame(10, "1h")
        close_ts = int((df["timestamp"].iloc[4] - pd.Timestamp("1970-01-01"))
                       // pd.Timedelta("1s"))  # = close of candle 3 / open of 4

        class _F:
            def get_ohlcv(self, pair, tf, limit=500):
                return df.copy()

        g = Grader(decision_maker=None, data_fetcher=_F(), store=_FakeStore())
        after = g._path_after("X", "1h", close_ts, k=3)
        assert after is not None
        assert after["timestamp"].iloc[0] == df["timestamp"].iloc[4]


class _FakeStore:
    pass


class TestNewsRagWiring:
    def test_headlines_flow_from_store_to_news_run(self, tmp_path, monkeypatch):
        """A4: decide() must pass stored headline titles into news.run."""
        import time as _time
        import brain.decision_maker as bdm
        from persistence import Store

        store = Store(str(tmp_path / "rag.db"))
        store.add_news_item(item_id="n1", source="coindesk", title="BTC ETF inflows surge",
                            body="", url="u1", published_ts=_time.time() - 3600, assets=["BTC"])
        store.add_news_item(item_id="n2", source="theblock", title="Bitcoin funding flips negative",
                            body="", url="u2", published_ts=_time.time() - 7200, assets=["BTC"])

        monkeypatch.setattr(bdm, "POLICY_PATH", str(tmp_path / "brain.json"))
        dm = bdm.DecisionMaker(store=store)
        monkeypatch.setattr(dm.indicator, "decide",
                            lambda s, tf, **k: {"action": "skip", "confidence": 0.5})
        monkeypatch.setattr(dm.research, "decide",
                            lambda *a, **k: {"action": "skip", "confidence": 0.5})
        seen = {}

        def fake_run(pair, overall_json=None, headlines=None):
            seen["headlines"] = headlines
            return {"action": "SKIP", "confidence": 0.5}

        monkeypatch.setattr(dm.news, "run", fake_run)
        dm.decide("BTCUSDT", "4h", use_agents=("news",))
        # formatted with age + source tier (C2): "[1h ago] [tier-1] <title>"
        assert len(seen["headlines"]) == 2
        assert seen["headlines"][0].endswith("BTC ETF inflows surge")
        assert "[tier-1]" in seen["headlines"][0]      # coindesk = tier 1
        assert "[tier-1]" in seen["headlines"][1]      # theblock = tier 1
        assert "ago]" in seen["headlines"][0]
        store.close()

    def test_empty_corpus_appends_guard(self, monkeypatch):
        """A4: no headlines -> prompts carry the no-hallucination guard."""
        import agents.news_agent as na
        prompts = []
        monkeypatch.setattr(na, "_chat_json", lambda p: (prompts.append(p) or {
            "has_panic": False, "sentiment": "neutral", "confidence": 0.5,
            "top_headlines": []}))
        agent = na.NewsAgent()
        agent.scan_overall(headlines=None)
        assert "No verified recent headlines" in prompts[-1]

    def test_headlines_suppress_guard(self, monkeypatch):
        import agents.news_agent as na
        prompts = []
        monkeypatch.setattr(na, "_chat_json", lambda p: (prompts.append(p) or {
            "has_panic": False, "sentiment": "neutral", "confidence": 0.5,
            "top_headlines": []}))
        agent = na.NewsAgent()
        agent.scan_overall(headlines=["Real headline"])
        assert "No verified recent headlines" not in prompts[-1]
        assert "Real headline" in prompts[-1]


class TestLogic4Dominance:
    def _agent(self):
        from agents.research_agent import ResearchAgent
        return ResearchAgent()

    def test_roc_drives_score(self, monkeypatch):
        a = self._agent()
        score, details = a._logic4_btcdominance("4h", None, None,
                                                dom_level=55.0, dom_roc=0.03)
        # dom rising hard -> dom_score +1 -> alt-unfavorable tilt
        assert details["source"] == "dominance_roc"
        assert details["btcdom_score"] == 1.0
        assert score < 0

    def test_level_extreme_fallback(self):
        a = self._agent()
        _, d_hi = a._logic4_btcdominance("4h", None, None, dom_level=65.0, dom_roc=None)
        _, d_lo = a._logic4_btcdominance("4h", None, None, dom_level=35.0, dom_roc=None)
        assert d_hi["source"] == "dominance_level" and d_hi["btcdom_score"] == 0.3
        assert d_lo["btcdom_score"] == -0.3

    def test_news_fallback_then_unavailable(self, monkeypatch):
        import utils.macro_fetcher as mf
        monkeypatch.setattr(mf, "fetch_btc_dominance", lambda **k: None)
        a = self._agent()

        class _News:
            def run(self, pair):
                return {"pair_json": {"sentiment": "bullish", "confidence": 0.8}}

        _, d = a._logic4_btcdominance("4h", None, _News(), dom_level=None, dom_roc=None)
        assert d["source"] == "news_fallback" and d["btcdom_score"] > 0
        _, d2 = a._logic4_btcdominance("4h", None, None, dom_level=None, dom_roc=None)
        assert d2["source"] == "unavailable" and d2["btcdom_score"] == 0.0

    def test_no_btcdomusdt_fetch_remains(self):
        import inspect
        from agents.research_agent import ResearchAgent
        src = inspect.getsource(ResearchAgent._logic4_btcdominance)
        assert 'decide("BTCDOMUSDT"' not in src  # the dead spot fetch is gone


class TestMacroSnapshots:
    def test_snapshot_roc_roundtrip(self, tmp_path):
        from persistence import Store
        s = Store(str(tmp_path / "m.db"))
        s.add_macro_snapshot(1000.0, 50.0, 40.0)
        s.add_macro_snapshot(90000.0, 55.0, 60.0)
        prev = s.macro_snapshot_before(89999.0)
        assert prev["btc_dominance"] == 50.0
        assert s.macro_snapshot_before(500.0) is None
        s.close()


class TestUniverse:
    def test_symbols_swap(self):
        from cycle import SYMBOLS
        assert "SUIUSDT" in SYMBOLS and "LUNAUSDT" not in SYMBOLS
        assert len(SYMBOLS) == 48


class TestBrainDeadzoneV2:
    def _dm(self, tmp_path, monkeypatch):
        import brain.decision_maker as bdm
        monkeypatch.setattr(bdm, "POLICY_PATH", str(tmp_path / "brain.json"))
        dm = bdm.DecisionMaker(store=_FakeStore())
        # one weak buy voter (news, lowest weight) vs strong skips
        monkeypatch.setattr(dm.indicator, "decide",
                            lambda s, tf, **k: {"action": "skip", "confidence": 0.9})
        monkeypatch.setattr(dm.research, "decide",
                            lambda *a, **k: {"action": "skip", "confidence": 0.9})
        monkeypatch.setattr(dm.news, "run",
                            lambda *a, **k: {"action": "BUY", "confidence": 0.95})
        monkeypatch.setattr(dm, "_headlines_for", lambda s: None)
        return dm

    def test_weak_single_voter(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "BRAIN_DEADZONE_V2", False)
        dm = self._dm(tmp_path, monkeypatch)
        res = dm.decide("BTCUSDT", "4h", use_agents=("indicator", "research", "news"))
        assert res["final"]["action"] == "buy"            # v1: emitted despite ~12% conf
        assert res["final"]["confidence"] < config.BRAIN_MIN_CONF
        assert res["final"]["action_v2"] == "skip"        # shadow says skip

        monkeypatch.setattr(config, "BRAIN_DEADZONE_V2", True)
        res2 = dm.decide("BTCUSDT", "4h", use_agents=("indicator", "research", "news"))
        assert res2["final"]["action"] == "skip"

    def test_action_v2_persisted(self, tmp_path):
        from persistence import Store
        s = Store(str(tmp_path / "v2.db"))
        pid = s.record_prediction({
            "chartName": "X", "timeframe": "1h",
            "final": {"action": "buy", "confidence": 0.1, "score": 0.06, "action_v2": "skip"},
            "agents": {}, "policy": {}})
        assert s.get_prediction(pid)["final_action_v2"] == "skip"
        s.close()


class TestClaimGrading:
    def test_exactly_one_claim_wins(self, tmp_path):
        from persistence import Store
        s = Store(str(tmp_path / "c.db"))
        s.conn.execute("INSERT INTO predictions (id, pair, tf, created_ts) "
                       "VALUES ('p1','BTCUSDT','1h',1.0)")
        s.conn.commit()
        assert s.claim_grading("p1", "auto") is True
        assert s.claim_grading("p1", "manual") is False    # already claimed
        assert s.get_prediction("p1")["label_source"] == "auto"
        s.close()

    def test_manual_loser_takes_correction_path(self, tmp_path, monkeypatch):
        """If auto claims mid-callback, apply_manual_feedback must correct, not
        double-apply."""
        from persistence import Store
        from grader import Grader

        class _Agent:
            def __init__(self): self.calls = []
            def apply_reward(self, *a): self.calls.append(a)

        class _DM:
            def __init__(self):
                self.news = _Agent(); self.research = _Agent(); self.indicator = _Agent()
            def apply_brain_feedback(self, *a): pass

        s = Store(str(tmp_path / "r.db"))
        pid = s.record_prediction({
            "chartName": "X", "timeframe": "4h",
            "final": {"action": "buy", "confidence": 0.8, "score": 0.5},
            "agents": {"indicator": {"action": "buy", "confidence": 0.7,
                                     "raw": {"action": "buy", "details": {"blend": {"type1_share": 0.5}}}}},
            "policy": {}})
        g = Grader(_DM(), data_fetcher=None, store=s)
        # simulate auto winning the row first
        assert s.claim_grading(pid, "auto")
        s.record_reward(pid, "indicator", "buy", 1.0, source="auto")
        res = g.apply_manual_feedback(pid, "sell")
        assert res["status"] == "corrected"
        # correction = manual(-4) - auto(+1) = -5 applied once
        assert res["corrections"]["indicator"] == pytest.approx(-5.0)
        s.close()


class TestBroadcastFailure:
    def test_dev_send_failure_deactivates_session(self, tmp_path):
        import asyncio as aio
        from persistence import Store
        from telegram_app import Broadcaster

        class _BadBot:
            async def send_message(self, **kw):
                raise RuntimeError("tg down")

        s = Store(str(tmp_path / "b.db"))
        b = Broadcaster(_BadBot(), s, customer_chat_id=None, dev_chat_id=123)
        sid = aio.run(b.broadcast(pair="BTCUSDT", tf="1h", overall="buy", nwe="buy",
                                  conf=0.9, reason="nwe_ranging",
                                  decision={"agents": {}, "final": {}}))
        sess = s.get_session(sid)
        assert sess["active"] == 0    # no buttons ever existed -> not left active
        s.close()


class TestNweEventMode:
    def _df_beyond_band(self):
        # noisy flat price (constant would NaN-out RSI/StochRSI), then a hard
        # drop: crossing at bar n-3, still beyond band at n-2 and n-1
        n = 90
        rng = np.random.default_rng(3)
        close = 100.0 + rng.normal(0, 0.15, n)
        close[-3:] = 90.0 + rng.normal(0, 0.05, 3)
        ts = pd.date_range("2024-01-01", periods=n, freq="h")
        return pd.DataFrame({"timestamp": ts, "open": close, "high": close + 0.5,
                             "low": close - 0.5, "close": close, "volume": np.ones(n)})

    def test_state_mode_refires_event_mode_does_not(self):
        from agents import custom_indicators as ci
        df = ci.apply_nadaraya_watson_envelope(self._df_beyond_band())
        # state mode: last bar still beyond band -> fires
        state = ci.direct_signal_from_nwe(df)
        assert state and state["signal"] == "buy"
        # event mode: crossing happened 3 bars ago, last bar has no crossing
        event = ci.direct_signal_from_nwee(df)
        assert event is None or event["signal"] == "skip"

    def test_event_mode_fires_on_crossing_bar(self):
        from agents import custom_indicators as ci
        n = 60
        close = np.full(n, 100.0)
        close[-1] = 90.0    # crossing happens ON the last closed bar
        ts = pd.date_range("2024-01-01", periods=n, freq="h")
        df = pd.DataFrame({"timestamp": ts, "open": close, "high": close + 0.5,
                           "low": close - 0.5, "close": close, "volume": np.ones(n)})
        df = ci.apply_nadaraya_watson_envelope(df)
        event = ci.direct_signal_from_nwee(df)
        assert event and event["signal"] == "buy"

    def test_flag_switches_collector(self, monkeypatch):
        import agents.indicator_agent as ia
        df = self._df_beyond_band()
        agent = ia.IndicatorAgent()

        monkeypatch.setattr(config, "NWE_EVENT_MODE", False)
        state_dec = agent.decide("X", "1h", ohlcv=df.copy(), log=False)
        state_names = {d["name"]: d["signal"] for d in state_dec.details["direct_signals"]}
        assert state_names.get("nwe") == "buy"

        monkeypatch.setattr(config, "NWE_EVENT_MODE", True)
        event_dec = agent.decide("X", "1h", ohlcv=df.copy(), log=False)
        event_names = {d["name"]: d["signal"] for d in event_dec.details["direct_signals"]}
        assert event_names.get("nwe") in (None, "skip")
