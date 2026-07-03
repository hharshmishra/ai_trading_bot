"""Phase 4 verification: signal gate + cascade fix, the cycle runner, and the
Telegram broadcaster + callback->grader wiring (fake bot, no live tokens)."""
from __future__ import annotations

import asyncio
import sys
import types
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
try:
    import pandas_ta  # noqa: F401
except Exception:  # pragma: no cover
    sys.modules.setdefault("pandas_ta", types.ModuleType("pandas_ta"))


def make_decision(pair="BTCUSDT", tf="4h", action="buy", conf=0.9, nwe=None):
    direct = [{"name": "nwe", "signal": nwe, "confidence": 0.9}] if nwe else []
    return {
        "chartName": pair, "timeframe": tf,
        "agents": {
            "indicator": {"action": action, "confidence": conf,
                          "raw": {"action": action, "details": {
                              "direct_signals": direct,
                              "blend": {"type1_share": 0.6, "type2_share": 0.4, "fired_direct": "nwe"}}}},
            "research": {"action": action, "confidence": 0.6,
                         "raw": {"action": action, "rl": {"feats": [0.0] * 10, "action_idx": 2}}},
            "news": {"action": "BUY", "confidence": 0.6,
                     "raw": {"action": "BUY", "rl": {"features": [0.0] * 5, "action_idx": 2}}},
        },
        "final": {"action": action, "confidence": conf, "score": 0.5},
        "policy": {"weights": {"indicator": 0.6, "research": 0.3, "news": 0.1}},
    }


# --------------------------- signals (pure) ------------------------------- #
def test_timeframes_due_cascade():
    """timeframes_due speaks UTC (correctness v3): Binance closes 4h candles at
    UTC 0/4/8/12/16/20, 1d at UTC 00:00, 1w Monday UTC 00:00."""
    from signals import timeframes_due
    assert timeframes_due(datetime(2024, 1, 2, 3, 0)) == ["1h"]                    # Tue 03:00 UTC
    assert timeframes_due(datetime(2024, 1, 2, 4, 0)) == ["4h", "1h"]              # Tue 04:00 UTC
    assert timeframes_due(datetime(2024, 1, 2, 0, 0)) == ["1d", "4h", "1h"]        # Tue 00:00 UTC
    assert timeframes_due(datetime(2024, 1, 1, 0, 0)) == ["1w", "1d", "4h", "1h"]  # Mon 00:00 UTC


def test_timeframes_due_ist_to_utc_mapping():
    """Scheduler tick is IST :30 == the UTC :00 boundary: IST 09:30 -> UTC 04:00
    (4h due); Monday IST 05:30 -> Monday UTC 00:00 (full cascade); IST 00:30 ->
    UTC 19:00 (1h only — the old IST-hour cascade wrongly fired 4h+1d here)."""
    from datetime import timezone
    from zoneinfo import ZoneInfo
    from signals import timeframes_due
    ist = ZoneInfo("Asia/Kolkata")

    dt = datetime(2024, 1, 2, 9, 30, tzinfo=ist).astimezone(timezone.utc)
    assert timeframes_due(dt) == ["4h", "1h"]

    dt = datetime(2024, 1, 1, 5, 30, tzinfo=ist).astimezone(timezone.utc)   # Monday
    assert timeframes_due(dt) == ["1w", "1d", "4h", "1h"]

    dt = datetime(2024, 1, 2, 0, 30, tzinfo=ist).astimezone(timezone.utc)
    assert timeframes_due(dt) == ["1h"]


def test_signal_gate_rules():
    from signals import should_emit_signal
    # 1h emits ONLY on NWE direct
    assert should_emit_signal(make_decision(tf="1h", conf=0.99, nwe=None))[0] is False
    emit, overall, *_ = should_emit_signal(make_decision(tf="1h", conf=0.1, nwe="buy"))
    assert emit and overall == "buy"
    # other TFs: conf>=0.80 emits
    assert should_emit_signal(make_decision(tf="4h", conf=0.9, nwe=None))[0] is True
    assert should_emit_signal(make_decision(tf="4h", conf=0.4, nwe=None))[0] is False
    # NWE overrides a conflicting confident call
    emit, overall, nwe, conf, reason = should_emit_signal(
        make_decision(tf="4h", action="buy", conf=0.9, nwe="sell"))
    assert emit and overall == "sell" and reason == "nwe_direct"


# ----------------------------- cycle runner ------------------------------- #
def test_run_cycle_records_and_gates(tmp_path):
    from persistence import Store
    from cycle import run_cycle

    store = Store(str(tmp_path / "c.db"))
    df = pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=10, freq="4h"),
                       "open": 1.0, "high": 1.0, "low": 1.0, "close": 100.0, "volume": 1.0})
    fetcher = SimpleNamespace(get_ohlcv=lambda s, tf, limit=500: df.copy())
    # A emits (conf 0.9), B does not (conf 0.4)
    dm = SimpleNamespace(
        indicator=None, news=None, research=None,
        decide=lambda sym, tf, ua, ctx: make_decision(sym, tf, "buy", 0.9 if sym == "AUSDT" else 0.4))

    broadcasts = []

    async def fake_broadcast(**kw):
        broadcasts.append(kw["pair"])
        return f"sess-{kw['pair']}"

    summary = asyncio.run(run_cycle(
        ["4h"], dm=dm, data_fetcher=fetcher, broadcast=fake_broadcast,
        symbols=["AUSDT", "BUSDT"], store=store, build_context=lambda *a, **k: None))

    assert summary["analyzed"] == 2 and summary["emitted"] == 1
    assert broadcasts == ["AUSDT"]
    with store._lock:
        rows = {r["pair"]: r for r in store.conn.execute(
            "SELECT pair, emitted, session_id, entry_price, grade_due_ts FROM predictions").fetchall()}
    assert len(rows) == 2
    assert rows["AUSDT"]["emitted"] == 1 and rows["AUSDT"]["session_id"] == "sess-AUSDT"
    assert rows["BUSDT"]["emitted"] == 0
    assert rows["AUSDT"]["entry_price"] == 100.0 and rows["AUSDT"]["grade_due_ts"] is not None
    store.close()


# ------------------- broadcaster + callback (fake bot) -------------------- #
class FakeMsg:
    def __init__(self, mid):
        self.message_id = mid


class FakeBot:
    def __init__(self):
        self.sent, self.edited, self._mid = [], [], 0

    async def send_message(self, **kw):
        self._mid += 1
        self.sent.append(kw)
        return FakeMsg(self._mid)

    async def edit_message_reply_markup(self, **kw):
        self.edited.append(kw)


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


def test_broadcaster_creates_and_supersedes_session(tmp_path):
    from persistence import Store
    from telegram_app import Broadcaster

    async def body():
        store = Store(str(tmp_path / "b.db"))
        bot = FakeBot()
        bc = Broadcaster(bot, store, customer_chat_id=111, dev_chat_id=222)
        dec = make_decision()
        sid = await bc.broadcast(pair="BTCUSDT", tf="4h", overall="buy", nwe="buy",
                                 conf=0.9, reason="nwe_direct", decision=dec)
        assert store.get_active_session("BTCUSDT", "4h")["id"] == sid
        assert len(bot.sent) == 2                                  # customer + dev
        sid2 = await bc.broadcast(pair="BTCUSDT", tf="4h", overall="sell", nwe="sell",
                                  conf=0.9, reason="nwe_direct", decision=dec)
        assert store.get_session(sid)["active"] == 0               # superseded
        assert store.get_active_session("BTCUSDT", "4h")["id"] == sid2
        store.close()

    asyncio.run(body())


def test_callback_applies_feedback_via_grader(tmp_path):
    from persistence import Store
    from grader import Grader
    from telegram_app import Broadcaster, handle_callback

    async def body():
        store = Store(str(tmp_path / "cb.db"))
        dm = FakeDM()
        grader = Grader(dm, data_fetcher=None, store=store)
        bc = Broadcaster(FakeBot(), store, 111, 222)
        pid = store.record_prediction(make_decision(), candle_close_ts=100.0, entry_price=100.0,
                                      horizon_k=2, grade_due_ts=1.0)
        sid = store.create_session(pair="BTCUSDT", tf="4h", prediction_id=pid,
                                   dev_chat_id=222, dev_msg_id=9)
        bd = {"store": store, "grader": grader, "broadcaster": bc}
        ctx = SimpleNamespace(application=SimpleNamespace(bot_data=bd))

        class CQ:
            def __init__(self, data):
                self.data = data; self.answers = []

            async def answer(self, text="", show_alert=False):
                self.answers.append(text)

        # set outcome, then reward -> grader applies manual feedback
        await handle_callback(SimpleNamespace(callback_query=CQ(f"{sid}|OUTCOME|buy")), ctx)
        assert store.get_session(sid)["true_outcome"] == "buy"
        await handle_callback(SimpleNamespace(callback_query=CQ(f"{sid}|REWARD|auto")), ctx)

        assert store.get_prediction(pid)["label_source"] == "manual"
        assert store.get_session(sid)["active"] == 0
        assert dm.news.calls and dm.research.calls and dm.indicator.calls   # trained from stored payload
        store.close()

    asyncio.run(body())
