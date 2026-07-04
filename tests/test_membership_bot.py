"""Membership P3: Bot D handlers — storefront, payment flows, doors, admin —
with the FakeBot pattern (no live Telegram)."""
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
from membership.payments import RazorpayLinks, TronWatcher
from membership.plans import SKUS, plans_text
from membership.store import SubsStore
from membership_fakes import (FakeBot, FakeCQ, FakeJoinRequest, FakeRzpHttp,
                              FakeTronHttp, mk_cmd_update, mk_ctx)

T0 = 1_800_000_000.0
CHANNEL = -100123


@pytest.fixture
def env(tmp_path):
    subs = SubsStore(str(tmp_path / "subs.db"))
    bot = FakeBot()
    bd = {"subs": subs, "channel_id": CHANNEL,
          "rzp": RazorpayLinks("kid", "sec", http=FakeRzpHttp()),
          "tron": TronWatcher("TWallet", http=FakeTronHttp("TWallet"))}
    yield subs, bot, bd
    subs.close()


def test_plans_text_covers_all_skus_both_currencies():
    t = plans_text()
    for s in SKUS.values():
        assert f"₹{s.inr}" in t and f"{s.usdt:g} USDT" in t


def test_cmd_plans_renders_keyboard(env):
    subs, bot, bd = env
    upd = mk_cmd_update()
    asyncio.run(mbot.cmd_plans(upd, mk_ctx(bd, bot)))
    r = upd.message.replies[0]
    buttons = [b for row in r["reply_markup"].inline_keyboard for b in row]
    assert len(buttons) == len(SKUS)


def test_sku_callback_offers_rails(env):
    subs, bot, bd = env
    cq = FakeCQ("sub|SIG-30")
    asyncio.run(mbot.handle_membership_callback(
        SimpleNamespace(callback_query=cq), mk_ctx(bd, bot)))
    r = cq.message.replies[0]
    labels = [b.text for row in r["reply_markup"].inline_keyboard for b in row]
    assert any("UPI" in x for x in labels) and any("USDT" in x for x in labels)


def test_pay_usdt_creates_fingerprinted_pending(env):
    subs, bot, bd = env
    cq = FakeCQ("pay|SIG-30|usdt", uid=55)
    asyncio.run(mbot.handle_membership_callback(
        SimpleNamespace(callback_query=cq), mk_ctx(bd, bot)))
    p = subs.pending_payments(method="tron")[0]
    assert p["user_id"] == 55 and p["fingerprint"] is not None
    msg = cq.message.replies[0]["text"]
    assert f"{p['amount']:.3f} USDT" in msg and "TWallet" in msg


def test_pay_inr_creates_link_and_ref(env):
    subs, bot, bd = env
    cq = FakeCQ("pay|SIG-30|inr", uid=56)
    asyncio.run(mbot.handle_membership_callback(
        SimpleNamespace(callback_query=cq), mk_ctx(bd, bot)))
    p = subs.pending_payments(method="razorpay")[0]
    assert p["ref"].startswith("plink_")
    assert "rzp.io" in cq.message.replies[0]["text"]


def test_pay_inr_unconfigured_offers_usdt(env, monkeypatch):
    subs, bot, bd = env
    monkeypatch.delenv("RAZORPAY_KEY_ID", raising=False)
    monkeypatch.delenv("RAZORPAY_KEY_SECRET", raising=False)
    bd["rzp"] = RazorpayLinks(http=FakeRzpHttp())          # no keys
    cq = FakeCQ("pay|SIG-30|inr", uid=57)
    asyncio.run(mbot.handle_membership_callback(
        SimpleNamespace(callback_query=cq), mk_ctx(bd, bot)))
    assert "USDT" in cq.answers[0]
    assert subs.pending_payments() == []                   # nothing dangling


def test_activate_and_welcome_mints_invite_and_dm(env):
    subs, bot, bd = env
    subs.touch_user(60, now_ts=T0)
    p = subs.create_pending_payment(60, "BUN-30", "INR", "razorpay", now_ts=T0)
    ok = asyncio.run(mbot.activate_and_welcome(bot, subs, p, CHANNEL, now_ts=T0))
    assert ok
    assert bot.invites[0]["chat_id"] == CHANNEL and bot.invites[0]["member_limit"] == 1
    dm = bot.sent[0]["text"]
    assert "inv1" in dm and "referral code" in dm and "Pro commands" in dm
    # idempotent: consumed payment activates nothing
    assert not asyncio.run(mbot.activate_and_welcome(bot, subs, p, CHANNEL, now_ts=T0))
    assert len(bot.sent) == 1


def test_pro_only_sku_mints_no_invite(env):
    subs, bot, bd = env
    subs.touch_user(61, now_ts=T0)
    p = subs.create_pending_payment(61, "PRO-30", "INR", "razorpay", now_ts=T0)
    asyncio.run(mbot.activate_and_welcome(bot, subs, p, CHANNEL, now_ts=T0))
    assert bot.invites == []


def test_join_request_gate(env):
    subs, bot, bd = env
    subs.grant(70, 30, "signals", now_ts=T0)
    req = FakeJoinRequest(70)
    asyncio.run(mbot.handle_join_request(
        SimpleNamespace(chat_join_request=req), mk_ctx(bd, bot)))
    assert req.approved and not req.declined

    stranger = FakeJoinRequest(71)
    asyncio.run(mbot.handle_join_request(
        SimpleNamespace(chat_join_request=stranger), mk_ctx(bd, bot)))
    assert stranger.declined and not stranger.approved
    assert bot.sent and bot.sent[-1]["chat_id"] == 71      # plans deep link DM


def test_admin_gate_and_grant(env, monkeypatch):
    subs, bot, bd = env
    monkeypatch.setattr(config, "ADMIN_USER_IDS", frozenset({999}))
    ctx = mk_ctx(bd, bot, args=["80", "7", "signals"])

    outsider = mk_cmd_update(uid=42)
    asyncio.run(mbot.cmd_grant(outsider, ctx))
    assert outsider.message.replies == []                  # silently ignored
    assert not subs.is_active(80, "signals", now_ts=T0)

    admin = mk_cmd_update(uid=999)
    asyncio.run(mbot.cmd_grant(admin, ctx))
    assert subs.is_active(80, "signals")
    assert bot.invites and bot.sent[-1]["chat_id"] == 80   # invite DM to grantee
    assert "Granted" in admin.message.replies[0]["text"]


def test_paid_command_matches_on_chain(env):
    subs, bot, bd = env
    subs.touch_user(90, now_ts=T0)
    p = subs.create_pending_payment(90, "SIG-30", "USDT", "tron", now_ts=T0)
    bd["tron"].http.transfers = [{"to": "TWallet",
                                  "value": str(int(round(p["fingerprint"] * 1e6))),
                                  "transaction_id": "txZ",
                                  "block_timestamp": int((T0 + 90) * 1000)}]
    upd = mk_cmd_update(uid=90)
    asyncio.run(mbot.cmd_paid(upd, mk_ctx(bd, bot)))
    assert subs.is_active(90, "signals")
    assert bot.invites                                     # welcome flow ran


def test_revoke_kicks_from_channel_immediately(env, monkeypatch):
    """The hourly kicker only sweeps ACTIVE rows — /revoke must remove channel
    access itself, or the revoked member keeps reading signals."""
    subs, bot, bd = env
    monkeypatch.setattr(config, "ADMIN_USER_IDS", frozenset({999}))
    subs.grant(85, 30, "signals", now_ts=T0)

    admin = mk_cmd_update(uid=999)
    asyncio.run(mbot.cmd_revoke(admin, mk_ctx(bd, bot, args=["85", "signals"])))
    assert not subs.is_active(85, "signals")
    assert (CHANNEL, 85) in bot.banned and (CHANNEL, 85) in bot.unbanned
    assert "removed" in admin.message.replies[0]["text"]

    # pro revoke touches no channel
    subs.grant(86, 30, "pro", now_ts=T0)
    asyncio.run(mbot.cmd_revoke(mk_cmd_update(uid=999), mk_ctx(bd, bot, args=["86", "pro"])))
    assert (CHANNEL, 86) not in bot.banned
