"""Membership P1: SubsStore — activation, renewal math, grace, usage,
lifecycle queries, referrals, admin ops. All clocks injected."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from membership.plans import SKUS
from membership.store import DAY_S, SubsStore

T0 = 1_800_000_000.0        # fixed epoch base for every test


@pytest.fixture
def subs(tmp_path):
    s = SubsStore(str(tmp_path / "subs.db"))
    yield s
    s.close()


def _pay(subs, uid=101, sku="SIG-30", currency="INR", method="razorpay", now=T0):
    subs.touch_user(uid, now_ts=now)
    p = subs.create_pending_payment(uid, sku, currency, method, now_ts=now)
    rows = subs.mark_paid(p["id"], ref="r1", now_ts=now)
    return p, rows


class TestPayments:
    def test_pending_inr_amount_is_sku_price(self, subs):
        p = subs.create_pending_payment(101, "SIG-30", "INR", "razorpay", now_ts=T0)
        assert p["amount"] == 599 and p["fingerprint"] is None

    def test_usdt_fingerprints_unique_and_suffixed(self, subs):
        a = subs.create_pending_payment(101, "SIG-30", "USDT", "tron", now_ts=T0)
        b = subs.create_pending_payment(102, "SIG-30", "USDT", "tron", now_ts=T0)
        base = SKUS["SIG-30"].usdt
        for p in (a, b):
            assert base + 0.100 < p["fingerprint"] < base + 1.0
        assert a["fingerprint"] != b["fingerprint"]

    def test_mark_paid_is_consume_once(self, subs):
        p, rows = _pay(subs)
        assert len(rows) == 1 and rows[0]["product"] == "signals"
        assert subs.mark_paid(p["id"], now_ts=T0) == []      # double poll -> no-op

    def test_bundle_activates_both_products(self, subs):
        _, rows = _pay(subs, sku="BUN-30")
        assert {r["product"] for r in rows} == {"signals", "pro"}
        assert subs.is_active(101, "signals", now_ts=T0)
        assert subs.is_active(101, "pro", now_ts=T0)


class TestRenewalMath:
    def test_renewal_extends_from_expiry_not_now(self, subs):
        _pay(subs, now=T0)                                    # expires T0+30d
        _, rows = _pay(subs, now=T0 + 10 * DAY_S)             # renew 20d early
        assert rows[0]["expires_ts"] == pytest.approx(T0 + 60 * DAY_S)

    def test_renewal_after_lapse_extends_from_now(self, subs):
        _pay(subs, now=T0)
        _, rows = _pay(subs, now=T0 + 40 * DAY_S)             # 10d after expiry
        assert rows[0]["expires_ts"] == pytest.approx(T0 + 70 * DAY_S)

    def test_renewal_resets_lifecycle_flags(self, subs):
        _pay(subs, now=T0)
        subs.mark_reminded(101, "signals", 2)
        subs.mark_kicked(101, "signals")
        _pay(subs, now=T0 + 40 * DAY_S)
        assert subs.is_active(101, "signals", now_ts=T0 + 41 * DAY_S)
        assert subs.due_reminders(now_ts=T0 + 41 * DAY_S) == []   # stage reset, not due


class TestAccess:
    def test_is_active_boundaries_with_grace(self, subs):
        _pay(subs, now=T0)
        exp = T0 + 30 * DAY_S
        grace = config.MEMBERSHIP_GRACE_HOURS * 3600
        assert subs.is_active(101, "signals", now_ts=exp - 1)
        assert subs.is_active(101, "signals", now_ts=exp + grace - 1)   # grace holds
        assert not subs.is_active(101, "signals", now_ts=exp + grace + 1)
        assert not subs.is_active(101, "pro", now_ts=T0)                # other product

    def test_revoke_kills_access(self, subs):
        _pay(subs)
        assert subs.revoke(101, "signals")
        assert not subs.is_active(101, "signals", now_ts=T0)

    def test_usage_counter_and_ist_rollover(self, subs):
        assert subs.bump_usage(101, now_ts=T0) == 1
        assert subs.bump_usage(101, now_ts=T0) == 2
        assert subs.bump_usage(101, now_ts=T0 + DAY_S) == 1   # next IST day resets


class TestLifecycleQueries:
    def test_reminder_stages_and_dedup(self, subs):
        _pay(subs, now=T0)
        exp = T0 + 30 * DAY_S
        assert subs.due_reminders(now_ts=T0 + 5 * DAY_S) == []
        due = subs.due_reminders(now_ts=exp - 2.5 * DAY_S)
        assert len(due) == 1 and due[0][1] == 1               # T-3d stage
        subs.mark_reminded(101, "signals", 1)
        assert subs.due_reminders(now_ts=exp - 2.4 * DAY_S) == []       # deduped
        due = subs.due_reminders(now_ts=exp - 0.5 * DAY_S)
        assert len(due) == 1 and due[0][1] == 2               # T-1d stage
        subs.mark_reminded(101, "signals", 2)
        assert subs.due_reminders(now_ts=exp - 0.1 * DAY_S) == []

    def test_kick_only_past_grace_and_winback_once(self, subs):
        _pay(subs, now=T0)
        exp = T0 + 30 * DAY_S
        grace = config.MEMBERSHIP_GRACE_HOURS * 3600
        assert subs.due_kicks(now_ts=exp + grace - 60) == []
        due = subs.due_kicks(now_ts=exp + grace + 60)
        assert len(due) == 1
        subs.mark_kicked(101, "signals")
        assert subs.due_kicks(now_ts=exp + grace + 120) == []
        assert subs.due_winbacks(now_ts=exp + 6 * DAY_S) == []
        wb = subs.due_winbacks(now_ts=exp + 7 * DAY_S + 60)
        assert len(wb) == 1
        subs.mark_winback_sent(101, "signals")
        assert subs.due_winbacks(now_ts=exp + 8 * DAY_S) == []


class TestReferrals:
    def test_referral_rules(self, subs):
        a = subs.touch_user(1, "alice", now_ts=T0)
        subs.touch_user(2, "bob", now_ts=T0)
        assert not subs.note_referral(1, a["referral_code"])          # self
        assert not subs.note_referral(2, "NOPE")                      # unknown code
        assert subs.note_referral(2, a["referral_code"])
        assert not subs.note_referral(2, a["referral_code"])          # first wins, once

    def test_first_payment_credits_both_once(self, subs):
        a = subs.touch_user(1, "alice", now_ts=T0)
        _pay(subs, uid=1, now=T0)                                     # alice active 30d
        subs.touch_user(2, "bob", now_ts=T0)
        subs.note_referral(2, a["referral_code"])
        _, rows = _pay(subs, uid=2, now=T0)                           # bob's first payment
        bonus = config.REFERRAL_BONUS_DAYS * DAY_S
        assert rows[0]["expires_ts"] == pytest.approx(T0 + 30 * DAY_S)  # pre-bonus row
        assert subs.is_active(2, "signals", now_ts=T0 + 36 * DAY_S)     # 30+7d holds
        # alice extended too
        assert subs.is_active(1, "signals", now_ts=T0 + 30 * DAY_S + bonus - 60)
        # second payment: NO second bonus
        _pay(subs, uid=2, now=T0 + 1)
        assert not subs.is_active(2, "signals", now_ts=T0 + (30 + 7 + 30 + 7 + 1) * DAY_S)

    def test_referral_rejected_after_first_payment(self, subs):
        a = subs.touch_user(1, now_ts=T0)
        subs.touch_user(2, now_ts=T0)
        _pay(subs, uid=2, now=T0)
        assert not subs.note_referral(2, a["referral_code"])


class TestAdmin:
    def test_grant_and_stats(self, subs):
        subs.grant(7, 3, "signals", now_ts=T0)
        assert subs.is_active(7, "signals", now_ts=T0 + 2 * DAY_S)
        _pay(subs, uid=8, sku="PRO-30", now=T0)
        s = subs.stats(now_ts=T0)
        assert s["active"] == {"signals": 1, "pro": 1}
        assert s["expiring_7d"] == 1                                  # the 3d grant
        assert s["revenue_30d"] == {"INR": 499}                       # admin grant excluded
