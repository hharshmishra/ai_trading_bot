"""Membership P2: Razorpay client, TronGrid watcher, fingerprint matcher —
offline via injectable http fakes."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from membership.payments import RazorpayLinks, TronWatcher, match_transfers
from membership_fakes import FakeRzpHttp, FakeTronHttp

T0 = 1_800_000_000.0
WALLET = "TTestWallet123"


class TestRazorpay:
    def test_create_link_posts_paise_and_notes(self):
        http = FakeRzpHttp()
        rzp = RazorpayLinks("kid", "sec", http=http)
        lid, url = rzp.create_link(599, "Signals · 30 days", 101, "SIG-30", now_ts=T0)
        body = http.posts[0]["json"]
        assert body["amount"] == 59900 and body["currency"] == "INR"
        assert body["notes"] == {"user_id": "101", "sku": "SIG-30"}
        assert body["expire_by"] == int(T0 + 15 * 60)
        assert http.posts[0]["auth"] == ("kid", "sec")
        assert lid.startswith("plink_") and url.endswith(lid)

    def test_link_status_transitions(self):
        http = FakeRzpHttp()
        rzp = RazorpayLinks("kid", "sec", http=http)
        lid, _ = rzp.create_link(199, "x", 1, "SIG-7", now_ts=T0)
        assert rzp.link_status(lid) == "created"
        http.statuses[lid] = "paid"
        assert rzp.link_status(lid) == "paid"

    def test_unconfigured_without_env(self, monkeypatch):
        monkeypatch.delenv("RAZORPAY_KEY_ID", raising=False)
        monkeypatch.delenv("RAZORPAY_KEY_SECRET", raising=False)
        assert not RazorpayLinks(http=FakeRzpHttp()).configured
        assert RazorpayLinks("k", "s", http=FakeRzpHttp()).configured


class TestTron:
    def test_incoming_parses_and_filters(self):
        http = FakeTronHttp(WALLET)
        http.transfers = [
            {"to": WALLET, "value": "7013000", "transaction_id": "tx1",
             "block_timestamp": int((T0 + 60) * 1000)},
            {"to": "SomeoneElse", "value": "7013000", "transaction_id": "tx2",
             "block_timestamp": int((T0 + 60) * 1000)},
        ]
        out = TronWatcher(WALLET, http=http).incoming(T0)
        assert len(out) == 1
        assert out[0]["amount"] == pytest.approx(7.013)
        assert out[0]["tx_id"] == "tx1" and out[0]["ts"] == pytest.approx(T0 + 60)

    def test_incoming_empty_on_transport_error(self):
        class Boom:
            def get(self, *a, **k):
                raise OSError("down")
        assert TronWatcher(WALLET, http=Boom()).incoming(T0) == []

    def test_unconfigured_without_wallet(self, monkeypatch):
        monkeypatch.delenv("TRON_WALLET_ADDRESS", raising=False)
        assert not TronWatcher(http=FakeTronHttp(WALLET)).configured


class TestMatcher:
    def _pending(self, pid, fp, created=T0):
        return {"id": pid, "fingerprint": fp, "created_ts": created}

    def _tx(self, tid, amount, ts=T0 + 60):
        return {"tx_id": tid, "amount": amount, "ts": ts}

    def test_exact_fingerprint_match(self):
        m = match_transfers([self._pending("p1", 7.013)], [self._tx("t1", 7.013)])
        assert m == [({"id": "p1", "fingerprint": 7.013, "created_ts": T0}, "t1")]

    def test_wrong_amount_ignored(self):
        assert match_transfers([self._pending("p1", 7.013)],
                               [self._tx("t1", 7.014), self._tx("t2", 7.0)]) == []

    def test_one_tx_settles_one_pending(self):
        pends = [self._pending("p1", 7.013, created=T0),
                 self._pending("p2", 7.013, created=T0 + 5)]
        m = match_transfers(pends, [self._tx("t1", 7.013)])
        assert len(m) == 1 and m[0][0]["id"] == "p1"          # oldest wins

    def test_tx_before_order_ignored(self):
        m = match_transfers([self._pending("p1", 7.013, created=T0)],
                            [self._tx("t1", 7.013, ts=T0 - 3600)])
        assert m == []
