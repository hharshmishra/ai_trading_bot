"""Payment rails — Razorpay Payment Links (INR/UPI) and TRON TRC-20 USDT.

Both clients take an injectable ``http`` module (anything with .get/.post à la
requests) so every test runs offline. Neither client owns polling cadence —
that lives in membership.jobs. Credentials are read from the environment at
CALL time (they are secrets; never cached at import).

No webhook server anywhere: Razorpay links are POLLED for status, TRON
transfers are matched by the unique 3-decimal amount fingerprint the store
assigned to the pending payment (see docs/subscription-deck.html §7).
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("membership.payments")

RZP_BASE = "https://api.razorpay.com/v1"
TRONGRID_BASE = "https://api.trongrid.io"
USDT_TRC20 = "TR7NHqjeKQxGTCi8q8ZY4pL8otSzgjLj6t"   # mainnet USDT contract
LINK_TTL_S = 15 * 60          # razorpay links expire after 15 min
TRON_TTL_S = 60 * 60          # USDT orders stay matchable for 60 min


def _requests():
    import requests
    return requests


class RazorpayLinks:
    def __init__(self, key_id: Optional[str] = None, key_secret: Optional[str] = None,
                 http=None):
        self._key_id = key_id
        self._key_secret = key_secret
        self.http = http or _requests()

    def _auth(self) -> Optional[Tuple[str, str]]:
        kid = self._key_id or os.getenv("RAZORPAY_KEY_ID")
        sec = self._key_secret or os.getenv("RAZORPAY_KEY_SECRET")
        if not kid or not sec:
            return None
        return (kid, sec)

    @property
    def configured(self) -> bool:
        return self._auth() is not None

    def create_link(self, amount_inr: float, description: str, user_id: int,
                    sku: str, now_ts: Optional[float] = None) -> Tuple[str, str]:
        """POST /payment_links -> (link_id, short_url). Raises on HTTP errors —
        the bot layer catches and offers the USDT rail instead."""
        now = now_ts if now_ts is not None else time.time()
        r = self.http.post(
            f"{RZP_BASE}/payment_links",
            auth=self._auth(),
            json={"amount": int(round(amount_inr * 100)),      # paise
                  "currency": "INR",
                  "description": description,
                  "expire_by": int(now + LINK_TTL_S),
                  "notes": {"user_id": str(user_id), "sku": sku}},
            timeout=15)
        r.raise_for_status()
        d = r.json()
        return d["id"], d["short_url"]

    def link_status(self, link_id: str) -> str:
        """'created' | 'paid' | 'expired' | 'cancelled' (razorpay statuses)."""
        r = self.http.get(f"{RZP_BASE}/payment_links/{link_id}",
                          auth=self._auth(), timeout=15)
        r.raise_for_status()
        return r.json().get("status", "created")


class TronWatcher:
    def __init__(self, wallet: Optional[str] = None, api_key: Optional[str] = None,
                 http=None):
        self._wallet = wallet
        self._api_key = api_key
        self.http = http or _requests()

    @property
    def wallet(self) -> Optional[str]:
        return self._wallet or os.getenv("TRON_WALLET_ADDRESS")

    @property
    def configured(self) -> bool:
        return bool(self.wallet)

    def incoming(self, since_ts: float) -> List[Dict[str, Any]]:
        """Confirmed inbound USDT transfers since ``since_ts``:
        [{'amount': 7.013, 'tx_id': ..., 'ts': ...}]. Empty list on any
        failure — the poller just tries again next tick."""
        wallet = self.wallet
        if not wallet:
            return []
        headers = {}
        key = self._api_key or os.getenv("TRONGRID_API_KEY")
        if key:
            headers["TRON-PRO-API-KEY"] = key
        try:
            r = self.http.get(
                f"{TRONGRID_BASE}/v1/accounts/{wallet}/transactions/trc20",
                params={"only_confirmed": "true", "only_to": "true", "limit": 50,
                        "min_timestamp": int(since_ts * 1000),
                        "contract_address": USDT_TRC20},
                headers=headers, timeout=15)
            r.raise_for_status()
            out = []
            for tx in r.json().get("data", []):
                if tx.get("to") != wallet:
                    continue
                out.append({"amount": int(tx["value"]) / 1e6,
                            "tx_id": tx.get("transaction_id"),
                            "ts": tx.get("block_timestamp", 0) / 1000.0})
            return out
        except Exception as e:
            logger.warning("trongrid fetch failed: %s", e)
            return []


def match_transfers(pendings: List[Dict[str, Any]],
                    transfers: List[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], str]]:
    """Match pending USDT payments to on-chain transfers by exact amount
    fingerprint (3-decimal, compared at integer millis to dodge float noise).
    One transfer settles exactly one pending; oldest pending wins a contested
    amount (fingerprints are unique among OPEN pendings by construction, so
    contests only arise from stale history)."""
    used = set()
    out = []
    for p in sorted(pendings, key=lambda x: x["created_ts"]):
        want = round((p.get("fingerprint") or 0) * 1000)
        if not want:
            continue
        for t in transfers:
            tid = t.get("tx_id")
            if tid in used:
                continue
            if round(t["amount"] * 1000) == want and t["ts"] >= p["created_ts"] - 60:
                used.add(tid)
                out.append((p, tid))
                break
    return out
