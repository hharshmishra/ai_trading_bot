"""Membership P4: payment poller + lifecycle sweep (reminders, kicks,
winbacks) + the full pay->kick->renew->rejoin integration, all with injected
clocks and fakes."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

import config
from membership import bot as mbot
from membership.jobs import TRON_GRACE_S, lifecycle_sweep_once, poll_payments_once
from membership.payments import RazorpayLinks, TronWatcher, TRON_TTL_S
from membership.store import DAY_S, SubsStore
from membership_fakes import FakeBot, FakeJoinRequest, FakeRzpHttp, FakeTronHttp, mk_ctx

T0 = 1_800_000_000.0
CHANNEL = -100123
GRACE_S = config.MEMBERSHIP_GRACE_HOURS * 3600


@pytest.fixture
def env(tmp_path):
    subs = SubsStore(str(tmp_path / "subs.db"))
    rzp_http, tron_http = FakeRzpHttp(), FakeTronHttp("TWallet")
    bd = {"subs": subs, "channel_id": CHANNEL,
          "rzp": RazorpayLinks("kid", "sec", http=rzp_http),
          "tron": TronWatcher("TWallet", http=tron_http)}
    yield subs, FakeBot(), bd, rzp_http, tron_http
    subs.close()


def _rzp_pending(subs, rzp, uid=101, sku="SIG-30", now=T0):
    subs.touch_user(uid, now_ts=now)
    p = subs.create_pending_payment(uid, sku, "INR", "razorpay", now_ts=now)
    lid, _ = rzp.create_link(1, "x", uid, sku, now_ts=now)
    subs.set_payment_ref(p["id"], lid)
    return p, lid


class TestPaymentPoller:
    def test_paid_link_activates_once(self, env):
        subs, bot, bd, rzp_http, _ = env
        p, lid = _rzp_pending(subs, bd["rzp"])
        rzp_http.statuses[lid] = "paid"
        assert asyncio.run(poll_payments_once(bd, bot, now_ts=T0 + 60)) == 1
        assert subs.is_active(101, "signals", now_ts=T0 + 60)
        # second poll: consumed payment, zero activations, zero extra DMs
        sent = len(bot.sent)
        assert asyncio.run(poll_payments_once(bd, bot, now_ts=T0 + 90)) == 0
        assert len(bot.sent) == sent

    def test_expired_link_marks_payment_expired(self, env):
        subs, bot, bd, rzp_http, _ = env
        p, lid = _rzp_pending(subs, bd["rzp"])
        rzp_http.statuses[lid] = "expired"
        asyncio.run(poll_payments_once(bd, bot, now_ts=T0 + 60))
        assert subs.pending_payments() == []
        assert not subs.is_active(101, "signals", now_ts=T0 + 60)
        # the buyer is told, and pointed back at the storefront
        dms = [m for m in bot.sent if m["chat_id"] == 101]
        assert len(dms) == 1
        assert "expired" in dms[0]["text"] and "/plans" in dms[0]["text"]

    def test_stale_pending_expires_by_age(self, env):
        subs, bot, bd, rzp_http, _ = env
        _rzp_pending(subs, bd["rzp"])
        asyncio.run(poll_payments_once(bd, bot, now_ts=T0 + 3600))    # > TTL+120
        assert subs.pending_payments() == []
        dms = [m for m in bot.sent if m["chat_id"] == 101]
        assert len(dms) == 1
        assert "expired" in dms[0]["text"] and "/plans" in dms[0]["text"]

    def test_tron_transfer_activates(self, env):
        subs, bot, bd, _, tron_http = env
        subs.touch_user(102, now_ts=T0)
        p = subs.create_pending_payment(102, "SIG-30", "USDT", "tron", now_ts=T0)
        tron_http.transfers = [{"to": "TWallet",
                                "value": str(int(round(p["fingerprint"] * 1e6))),
                                "transaction_id": "tx9",
                                "block_timestamp": int((T0 + 120) * 1000)}]
        assert asyncio.run(poll_payments_once(bd, bot, now_ts=T0 + 180)) == 1
        assert subs.is_active(102, "signals", now_ts=T0 + 180)
        assert not any("expired" in m["text"] for m in bot.sent)      # welcome only

    def test_tron_expiry_dm_mentions_paid_rescue(self, env):
        subs, bot, bd, *_ = env
        subs.touch_user(103, now_ts=T0)
        p = subs.create_pending_payment(103, "SIG-30", "USDT", "tron", now_ts=T0)
        asyncio.run(poll_payments_once(bd, bot, now_ts=T0 + TRON_TTL_S + TRON_GRACE_S + 60))
        assert subs.pending_payments() == []
        dms = [m for m in bot.sent if m["chat_id"] == 103]
        # the late-payer rescue hatch is advertised, with the exact amount
        assert len(dms) == 1 and "/paid" in dms[0]["text"]
        assert str(p["amount"]) in dms[0]["text"]

    def test_expiry_dm_failure_still_expires(self, env):
        subs, bot, bd, *_ = env
        subs.touch_user(104, now_ts=T0)
        subs.create_pending_payment(104, "SIG-30", "USDT", "tron", now_ts=T0)
        bot.raise_for_uids.add(104)                                   # buyer blocked the bot
        asyncio.run(poll_payments_once(bd, bot, now_ts=T0 + TRON_TTL_S + TRON_GRACE_S + 60))
        assert subs.pending_payments() == []                          # expiry unaffected


class TestLifecycleSweep:
    def test_reminders_sent_once_with_renew_buttons(self, env):
        subs, bot, bd, *_ = env
        subs.grant(201, 30, "signals", now_ts=T0)
        exp = T0 + 30 * DAY_S
        s = asyncio.run(lifecycle_sweep_once(bd, bot, now_ts=exp - 2 * DAY_S))
        assert s["reminded"] == 1
        assert "3 days" in bot.sent[-1]["text"]
        assert bot.sent[-1]["reply_markup"] is not None
        s = asyncio.run(lifecycle_sweep_once(bd, bot, now_ts=exp - 2 * DAY_S + 60))
        assert s["reminded"] == 0                                     # deduped

    def test_kick_bans_unbans_and_marks(self, env):
        subs, bot, bd, *_ = env
        subs.grant(202, 1, "signals", now_ts=T0)
        past = T0 + 1 * DAY_S + GRACE_S + 60
        s = asyncio.run(lifecycle_sweep_once(bd, bot, now_ts=past))
        assert s["kicked"] == 1
        assert bot.banned == [(CHANNEL, 202)] and bot.unbanned == [(CHANNEL, 202)]
        assert not subs.is_active(202, "signals", now_ts=past)
        s = asyncio.run(lifecycle_sweep_once(bd, bot, now_ts=past + 60))
        assert s["kicked"] == 0                                       # once

    def test_pro_kick_touches_no_channel(self, env):
        subs, bot, bd, *_ = env
        subs.grant(203, 1, "pro", now_ts=T0)
        asyncio.run(lifecycle_sweep_once(bd, bot, now_ts=T0 + DAY_S + GRACE_S + 60))
        assert bot.banned == []                                       # pro is bot-side only

    def test_winback_once_at_plus_7d(self, env):
        subs, bot, bd, *_ = env
        subs.grant(204, 1, "signals", now_ts=T0)
        exp = T0 + 1 * DAY_S
        asyncio.run(lifecycle_sweep_once(bd, bot, now_ts=exp + GRACE_S + 60))   # kick
        s = asyncio.run(lifecycle_sweep_once(bd, bot, now_ts=exp + 7 * DAY_S + 60))
        assert s["winback"] == 1
        s = asyncio.run(lifecycle_sweep_once(bd, bot, now_ts=exp + 8 * DAY_S))
        assert s["winback"] == 0

    def test_blocked_dm_does_not_abort_sweep(self, env):
        subs, bot, bd, *_ = env
        subs.grant(205, 30, "signals", now_ts=T0)
        subs.grant(206, 30, "signals", now_ts=T0)
        bot.raise_for_uids.add(205)                                   # 205 blocked the bot
        exp = T0 + 30 * DAY_S
        s = asyncio.run(lifecycle_sweep_once(bd, bot, now_ts=exp - 2 * DAY_S))
        assert s["reminded"] == 2                                     # both processed
        assert any(m["chat_id"] == 206 for m in bot.sent)             # 206 still got hers


def test_full_lifecycle_pay_kick_renew_rejoin(env):
    """The end-to-end story: pay -> active -> join approved -> expiry+grace ->
    kicked -> renew -> rejoin approved. Injected clocks only."""
    subs, bot, bd, rzp_http, _ = env
    ctx = mk_ctx(bd, bot)

    p, lid = _rzp_pending(subs, bd["rzp"], uid=301, sku="SIG-7", now=T0)
    rzp_http.statuses[lid] = "paid"
    asyncio.run(poll_payments_once(bd, bot, now_ts=T0 + 60))
    assert subs.is_active(301, "signals", now_ts=T0 + 60)

    req = FakeJoinRequest(301)
    asyncio.run(mbot.handle_join_request(SimpleNamespace(chat_join_request=req), ctx))
    assert req.approved

    lapsed = T0 + 7 * DAY_S + GRACE_S + 60
    asyncio.run(lifecycle_sweep_once(bd, bot, now_ts=lapsed))
    assert (CHANNEL, 301) in bot.banned
    req2 = FakeJoinRequest(301)
    asyncio.run(mbot.handle_join_request(SimpleNamespace(chat_join_request=req2), ctx))
    assert req2.declined                                              # lapsed = out

    p2, lid2 = _rzp_pending(subs, bd["rzp"], uid=301, sku="SIG-30", now=lapsed + 3600)
    rzp_http.statuses[lid2] = "paid"
    asyncio.run(poll_payments_once(bd, bot, now_ts=lapsed + 3660))
    req3 = FakeJoinRequest(301)
    asyncio.run(mbot.handle_join_request(SimpleNamespace(chat_join_request=req3), ctx))
    assert req3.approved                                              # welcome back
