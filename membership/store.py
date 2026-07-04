"""SubsStore — thread-safe SQLite store for subscriptions, payments and
pro-usage counters. Same pattern as persistence.Store (one connection, one
lock, WAL) but a SEPARATE database file: membership traffic must never contend
with the trading store's lock.

Time discipline: every method that reasons about time takes ``now_ts`` (epoch
seconds, defaults to time.time()) — the whole lifecycle is testable with
injected clocks, no sleeps (grader pattern).
"""
from __future__ import annotations

import os
import sqlite3
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import config
from membership.plans import SKUS

IST = timezone(timedelta(hours=5, minutes=30))
DAY_S = 86400.0


class FingerprintExhausted(Exception):
    """No free amount suffix for this SKU right now (>99 same-priced USDT
    orders live within the dedup window). The buyer is asked to retry."""


# The access boundary in ONE place: a subscription grants access iff it is
# 'active' and inside expiry + grace. is_active() and due_kicks() are exact
# complements of this predicate, so they can never disagree — including on a
# NULL expires_ts, which both treat as 0 (no access → kicked, no limbo row).
# Bind params in order: (grace_seconds, now_epoch).
_HAS_ACCESS = "status = 'active' AND COALESCE(expires_ts, 0) + ? > ?"
_PAST_ACCESS = "status = 'active' AND COALESCE(expires_ts, 0) + ? <= ?"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    user_id            INTEGER PRIMARY KEY,
    username           TEXT,
    referral_code      TEXT UNIQUE,
    referred_by        INTEGER,             -- user_id of referrer
    referral_credited  INTEGER DEFAULT 0,   -- bonus granted once, on first payment
    first_seen_ts      REAL
);
CREATE TABLE IF NOT EXISTS subscriptions (
    user_id          INTEGER NOT NULL,
    product          TEXT NOT NULL,          -- 'signals' | 'pro'
    started_ts       REAL,
    expires_ts       REAL,
    status           TEXT DEFAULT 'active',  -- active | kicked | revoked
    reminder_stage   INTEGER DEFAULT 0,      -- 0 none | 1 T-3d sent | 2 T-1d sent
    winback_sent     INTEGER DEFAULT 0,
    PRIMARY KEY (user_id, product)
);
CREATE TABLE IF NOT EXISTS payments (
    id           TEXT PRIMARY KEY,
    user_id      INTEGER NOT NULL,
    sku          TEXT NOT NULL,
    amount       REAL NOT NULL,
    currency     TEXT NOT NULL,              -- INR | USDT
    method       TEXT NOT NULL,              -- razorpay | tron | admin
    ref          TEXT,                       -- rzp link id / tron tx id
    fingerprint  REAL,                       -- USDT amount incl. unique suffix
    created_ts   REAL,
    paid_ts      REAL,
    status       TEXT DEFAULT 'pending'      -- pending | paid | expired
);
CREATE TABLE IF NOT EXISTS usage (
    user_id  INTEGER NOT NULL,
    day      TEXT NOT NULL,                  -- YYYY-MM-DD in IST
    queries  INTEGER DEFAULT 0,
    PRIMARY KEY (user_id, day)
);
CREATE INDEX IF NOT EXISTS idx_payments_status ON payments(status, paid_ts);
"""

# Additive migration (same discipline as persistence.Store): diff PRAGMA
# table_info and ADD COLUMN what's missing, so an existing subscriptions.db
# upgrades in place instead of needing a manual ALTER or a wipe.
_MIGRATION_COLS = {
    "subscriptions": [
        # 1 once the member has actually been removed from the channel; lets
        # /revoke's removal be retried by the hourly sweep if the live ban fails.
        ("channel_removed", "INTEGER DEFAULT 0"),
    ],
}


def _ist_day(now_ts: float) -> str:
    return datetime.fromtimestamp(now_ts, IST).strftime("%Y-%m-%d")


class SubsStore:
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or config.MEMBERSHIP_DB
        if self.db_path != ":memory:":
            os.makedirs(os.path.dirname(os.path.abspath(self.db_path)) or ".", exist_ok=True)
        self._lock = threading.Lock()
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        with self._lock:
            self.conn.execute("PRAGMA journal_mode=WAL;")
            self.conn.execute("PRAGMA synchronous=NORMAL;")
            self.conn.executescript(_SCHEMA)
            self._migrate()
            self.conn.commit()

    def _migrate(self) -> None:
        """Idempotent additive migration; caller holds the lock (init path)."""
        for table, cols in _MIGRATION_COLS.items():
            existing = {r["name"] for r in
                        self.conn.execute(f"PRAGMA table_info({table})").fetchall()}
            for name, ctype in cols:
                if name not in existing:
                    self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {ctype}")

    def close(self) -> None:
        with self._lock:
            self.conn.close()

    # ------------------------------------------------------------------ #
    # users / referrals
    # ------------------------------------------------------------------ #
    def touch_user(self, user_id: int, username: Optional[str] = None,
                   now_ts: Optional[float] = None) -> Dict[str, Any]:
        """Ensure a users row exists (issues the referral code) and return it."""
        now = now_ts if now_ts is not None else time.time()
        with self._lock:
            r = self.conn.execute("SELECT * FROM users WHERE user_id = ?", (user_id,)).fetchone()
            if r is None:
                code = f"BRX{user_id:X}"          # deterministic, unique per user
                self.conn.execute(
                    "INSERT INTO users (user_id, username, referral_code, first_seen_ts) "
                    "VALUES (?,?,?,?)", (user_id, username, code, now))
                self.conn.commit()
                r = self.conn.execute("SELECT * FROM users WHERE user_id = ?", (user_id,)).fetchone()
            elif username and r["username"] != username:
                self.conn.execute("UPDATE users SET username = ? WHERE user_id = ?",
                                  (username, user_id))
                self.conn.commit()
                r = self.conn.execute("SELECT * FROM users WHERE user_id = ?", (user_id,)).fetchone()
        return dict(r)

    def note_referral(self, user_id: int, code: str) -> bool:
        """Attach a referrer to a NEW user (no self-referrals, first code wins,
        never after the user already paid)."""
        with self._lock:
            u = self.conn.execute("SELECT * FROM users WHERE user_id = ?", (user_id,)).fetchone()
            ref = self.conn.execute("SELECT user_id FROM users WHERE referral_code = ?",
                                    (code,)).fetchone()
            if u is None or ref is None or ref["user_id"] == user_id or u["referred_by"]:
                return False
            paid = self.conn.execute(
                "SELECT 1 FROM payments WHERE user_id = ? AND status = 'paid' LIMIT 1",
                (user_id,)).fetchone()
            if paid:
                return False
            self.conn.execute("UPDATE users SET referred_by = ? WHERE user_id = ?",
                              (ref["user_id"], user_id))
            self.conn.commit()
            return True

    # ------------------------------------------------------------------ #
    # payments
    # ------------------------------------------------------------------ #
    def create_pending_payment(self, user_id: int, sku_code: str, currency: str,
                               method: str, now_ts: Optional[float] = None) -> Dict[str, Any]:
        """Register a pending payment. For USDT the TOTAL amount gets a tiny
        unique suffix (+0.001..+0.099) so an on-chain transfer maps to exactly
        one order.

        Uniqueness is keyed on the full amount in milli-USDT — the SAME
        quantity match_transfers compares — not on a reconstructed fractional
        part (which collided for half-integer base prices like 2.5/3.5/4.5).
        The dedup set spans every open pending AND any USDT payment created in
        the last 24h (paid or expired): a just-expired order's amount must not
        be reused while a late transfer for it could still arrive. Suffix
        1..99 keeps the overcharge under +0.1 USDT (was up to +0.999)."""
        sku = SKUS[sku_code]
        now = now_ts if now_ts is not None else time.time()
        pid = uuid.uuid4().hex
        amount = float(sku.inr if currency == "INR" else sku.usdt)
        fingerprint = None
        with self._lock:
            if currency == "USDT":
                base_millis = round(float(sku.usdt) * 1000)
                taken = {round(r["amount"] * 1000) for r in self.conn.execute(
                    "SELECT amount FROM payments WHERE currency = 'USDT' AND "
                    "(status = 'pending' OR created_ts > ?)", (now - DAY_S,)).fetchall()}
                chosen = next((base_millis + s for s in range(1, 100)
                               if (base_millis + s) not in taken), None)
                if chosen is None:
                    self.conn.commit()
                    raise FingerprintExhausted(sku_code)
                fingerprint = round(chosen / 1000.0, 3)
                amount = fingerprint
            self.conn.execute(
                "INSERT INTO payments (id, user_id, sku, amount, currency, method, "
                "fingerprint, created_ts, status) VALUES (?,?,?,?,?,?,?,?, 'pending')",
                (pid, user_id, sku_code, amount, currency, method, fingerprint, now))
            self.conn.commit()
            return dict(self.conn.execute("SELECT * FROM payments WHERE id = ?", (pid,)).fetchone())

    def set_payment_ref(self, payment_id: str, ref: str) -> None:
        with self._lock:
            self.conn.execute("UPDATE payments SET ref = ? WHERE id = ?", (ref, payment_id))
            self.conn.commit()

    def pending_payments(self, method: Optional[str] = None) -> List[Dict[str, Any]]:
        q = "SELECT * FROM payments WHERE status = 'pending'"
        args: Tuple = ()
        if method:
            q += " AND method = ?"
            args = (method,)
        with self._lock:
            return [dict(r) for r in self.conn.execute(q + " ORDER BY created_ts", args)]

    def rescuable_tron_payments(self, user_id: int,
                                now_ts: Optional[float] = None) -> List[Dict[str, Any]]:
        """This user's USDT orders still settleable by /paid: pending, OR
        expired within the last 24h (TTL lapsed before the transfer confirmed).
        """
        now = now_ts if now_ts is not None else time.time()
        with self._lock:
            return [dict(r) for r in self.conn.execute(
                "SELECT * FROM payments WHERE user_id = ? AND method = 'tron' "
                "AND (status = 'pending' OR (status = 'expired' AND created_ts > ?)) "
                "ORDER BY created_ts", (user_id, now - DAY_S)).fetchall()]

    def expire_payment(self, payment_id: str) -> None:
        with self._lock:
            self.conn.execute(
                "UPDATE payments SET status = 'expired' WHERE id = ? AND status = 'pending'",
                (payment_id,))
            self.conn.commit()

    def mark_paid(self, payment_id: str, ref: Optional[str] = None,
                  now_ts: Optional[float] = None,
                  allow_expired: bool = False) -> List[Dict[str, Any]]:
        """Activate a pending payment: CAS to 'paid', extend every product the
        SKU carries (renewal extends from max(now, current expiry) — no gap,
        no lost days), credit the referral bonus once on the user's FIRST paid
        payment. Returns the resulting subscription rows ([] if the payment
        was already consumed — double-poll safe).

        allow_expired lets the /paid rescue settle a USDT order whose TTL just
        lapsed before its on-chain transfer confirmed (the CAS then also
        matches status='expired')."""
        now = now_ts if now_ts is not None else time.time()
        statuses = "('pending', 'expired')" if allow_expired else "('pending')"
        with self._lock:
            cur = self.conn.execute(
                "UPDATE payments SET status = 'paid', ref = COALESCE(?, ref), paid_ts = ? "
                f"WHERE id = ? AND status IN {statuses}", (ref, now, payment_id))
            if cur.rowcount != 1:
                self.conn.commit()
                return []
            p = dict(self.conn.execute("SELECT * FROM payments WHERE id = ?", (payment_id,)).fetchone())
            sku = SKUS[p["sku"]]
            for product in sku.products:
                self._extend_locked(p["user_id"], product, sku.days * DAY_S, now)

            # first-paid referral: +REFERRAL_BONUS_DAYS to both sides, once
            u = self.conn.execute("SELECT * FROM users WHERE user_id = ?",
                                  (p["user_id"],)).fetchone()
            if (u and u["referred_by"] and not u["referral_credited"]):
                bonus = config.REFERRAL_BONUS_DAYS * DAY_S
                for product in sku.products:
                    self._extend_locked(p["user_id"], product, bonus, now)
                # referrer gets the bonus only on products they STILL hold access
                # to (inside expiry+grace) — never resurrect a past-grace sub or
                # reset its reminder/winback flags.
                grace = config.MEMBERSHIP_GRACE_HOURS * 3600.0
                for r in self.conn.execute(
                        "SELECT product, expires_ts FROM subscriptions "
                        "WHERE user_id = ? AND status = 'active'",
                        (u["referred_by"],)).fetchall():
                    if (r["expires_ts"] or 0) + grace > now:
                        self._extend_locked(u["referred_by"], r["product"], bonus, now)
                self.conn.execute("UPDATE users SET referral_credited = 1 WHERE user_id = ?",
                                  (p["user_id"],))
            self.conn.commit()
            # re-read AFTER the bonus so callers (welcome DM) see the true expiry
            rows = [dict(r) for r in self.conn.execute(
                "SELECT * FROM subscriptions WHERE user_id = ? AND product IN (%s)"
                % ",".join("?" * len(sku.products)),
                (p["user_id"], *sku.products)).fetchall()]
            return rows

    def _extend_locked(self, user_id: int, product: str, seconds: float,
                       now: float) -> Dict[str, Any]:
        r = self.conn.execute(
            "SELECT * FROM subscriptions WHERE user_id = ? AND product = ?",
            (user_id, product)).fetchone()
        if r is None:
            self.conn.execute(
                "INSERT INTO subscriptions (user_id, product, started_ts, expires_ts, status) "
                "VALUES (?,?,?,?, 'active')", (user_id, product, now, now + seconds))
        else:
            base = max(now, r["expires_ts"] or now)
            self.conn.execute(
                "UPDATE subscriptions SET expires_ts = ?, status = 'active', "
                "reminder_stage = 0, winback_sent = 0 WHERE user_id = ? AND product = ?",
                (base + seconds, user_id, product))
        return dict(self.conn.execute(
            "SELECT * FROM subscriptions WHERE user_id = ? AND product = ?",
            (user_id, product)).fetchone())

    # ------------------------------------------------------------------ #
    # access checks
    # ------------------------------------------------------------------ #
    def is_active(self, user_id: int, product: str,
                  now_ts: Optional[float] = None) -> bool:
        """Active = not revoked/kicked and inside expiry + grace. Grace means
        access CONTINUES for MEMBERSHIP_GRACE_HOURS past expiry (in-flight
        renewals never cause a wrongful lockout); the kick job uses the same
        boundary, so access and membership always agree."""
        now = now_ts if now_ts is not None else time.time()
        grace = config.MEMBERSHIP_GRACE_HOURS * 3600.0
        with self._lock:
            r = self.conn.execute(
                "SELECT 1 FROM subscriptions WHERE user_id = ? AND product = ? AND " + _HAS_ACCESS,
                (user_id, product, grace, now)).fetchone()
        return bool(r)

    def bump_usage(self, user_id: int, now_ts: Optional[float] = None) -> int:
        """Increment and return today's (IST) pro query count."""
        day = _ist_day(now_ts if now_ts is not None else time.time())
        with self._lock:
            self.conn.execute(
                "INSERT INTO usage (user_id, day, queries) VALUES (?,?,1) "
                "ON CONFLICT(user_id, day) DO UPDATE SET queries = queries + 1",
                (user_id, day))
            self.conn.commit()
            r = self.conn.execute("SELECT queries FROM usage WHERE user_id = ? AND day = ?",
                                  (user_id, day)).fetchone()
        return int(r["queries"])

    # ------------------------------------------------------------------ #
    # lifecycle queries (reminders / kicks / winbacks)
    # ------------------------------------------------------------------ #
    def due_reminders(self, now_ts: Optional[float] = None) -> List[Tuple[Dict[str, Any], int]]:
        """[(subscription, stage_to_send)]: stage 1 = T-3d, stage 2 = T-1d.
        A subscription skips straight to stage 2 if it was never reminded and
        is already inside T-1d."""
        now = now_ts if now_ts is not None else time.time()
        out = []
        with self._lock:
            rows = self.conn.execute(
                "SELECT * FROM subscriptions WHERE status = 'active' AND expires_ts > ?",
                (now,)).fetchall()
        for r in rows:
            left = r["expires_ts"] - now
            if left <= 1 * DAY_S and r["reminder_stage"] < 2:
                out.append((dict(r), 2))
            elif left <= 3 * DAY_S and r["reminder_stage"] < 1:
                out.append((dict(r), 1))
        return out

    def mark_reminded(self, user_id: int, product: str, stage: int) -> None:
        with self._lock:
            self.conn.execute(
                "UPDATE subscriptions SET reminder_stage = ? WHERE user_id = ? AND product = ? "
                "AND reminder_stage < ?", (stage, user_id, product, stage))
            self.conn.commit()

    def due_kicks(self, now_ts: Optional[float] = None) -> List[Dict[str, Any]]:
        """Active subscriptions past expiry + grace — remove access now."""
        now = now_ts if now_ts is not None else time.time()
        grace = config.MEMBERSHIP_GRACE_HOURS * 3600.0
        with self._lock:
            return [dict(r) for r in self.conn.execute(
                "SELECT * FROM subscriptions WHERE " + _PAST_ACCESS,
                (grace, now)).fetchall()]

    def mark_kicked(self, user_id: int, product: str) -> None:
        with self._lock:
            self.conn.execute(
                "UPDATE subscriptions SET status = 'kicked', channel_removed = 1 "
                "WHERE user_id = ? AND product = ?", (user_id, product))
            self.conn.commit()

    def due_channel_removals(self) -> List[Dict[str, Any]]:
        """Admin-revoked signals subscriptions not yet confirmed removed from the
        channel. /revoke tries an immediate ban; if it fails (transient Telegram
        error) the row is left channel_removed=0 and the hourly sweep retries
        here. Revoked rows never enter the winback flow (status stays 'revoked').
        """
        with self._lock:
            return [dict(r) for r in self.conn.execute(
                "SELECT * FROM subscriptions WHERE product = 'signals' "
                "AND status = 'revoked' AND channel_removed = 0").fetchall()]

    def mark_channel_removed(self, user_id: int, product: str) -> None:
        with self._lock:
            self.conn.execute(
                "UPDATE subscriptions SET channel_removed = 1 "
                "WHERE user_id = ? AND product = ?", (user_id, product))
            self.conn.commit()

    def gc(self, now_ts: Optional[float] = None) -> Dict[str, int]:
        """Prune unbounded history: usage rows older than 45 days and terminal
        (expired) payments older than 90 days. Paid/admin payments are kept
        (revenue ledger). Returns row counts deleted."""
        now = now_ts if now_ts is not None else time.time()
        cutoff_day = _ist_day(now - 45 * DAY_S)
        with self._lock:
            u = self.conn.execute("DELETE FROM usage WHERE day < ?", (cutoff_day,)).rowcount
            p = self.conn.execute(
                "DELETE FROM payments WHERE status = 'expired' AND created_ts < ?",
                (now - 90 * DAY_S,)).rowcount
            self.conn.commit()
        return {"usage": u, "payments": p}

    def due_winbacks(self, now_ts: Optional[float] = None) -> List[Dict[str, Any]]:
        """Kicked members 7+ days past expiry who never got the (single)
        winback nudge."""
        now = now_ts if now_ts is not None else time.time()
        with self._lock:
            return [dict(r) for r in self.conn.execute(
                "SELECT * FROM subscriptions WHERE status = 'kicked' AND winback_sent = 0 "
                "AND expires_ts + ? <= ?", (7 * DAY_S, now)).fetchall()]

    def mark_winback_sent(self, user_id: int, product: str) -> None:
        with self._lock:
            self.conn.execute(
                "UPDATE subscriptions SET winback_sent = 1 WHERE user_id = ? AND product = ?",
                (user_id, product))
            self.conn.commit()

    # ------------------------------------------------------------------ #
    # admin
    # ------------------------------------------------------------------ #
    def grant(self, user_id: int, days: int, product: str,
              now_ts: Optional[float] = None) -> Dict[str, Any]:
        now = now_ts if now_ts is not None else time.time()
        self.touch_user(user_id, now_ts=now)
        with self._lock:
            row = self._extend_locked(user_id, product, days * DAY_S, now)
            pid = uuid.uuid4().hex
            self.conn.execute(
                "INSERT INTO payments (id, user_id, sku, amount, currency, method, "
                "created_ts, paid_ts, status) VALUES (?,?,?,?,?,?,?,?, 'paid')",
                (pid, user_id, f"GRANT-{product}-{days}", 0.0, "INR", "admin", now, now))
            self.conn.commit()
        return row

    def revoke(self, user_id: int, product: str) -> bool:
        """Immediate no-access (status='revoked', expiry zeroed) + arm the
        channel-removal retry (channel_removed=0). cmd_revoke attempts a live
        ban; if it fails the hourly sweep finishes the job via
        due_channel_removals()."""
        with self._lock:
            cur = self.conn.execute(
                "UPDATE subscriptions SET status = 'revoked', expires_ts = 0, "
                "channel_removed = 0 WHERE user_id = ? AND product = ?",
                (user_id, product))
            self.conn.commit()
            return cur.rowcount == 1

    def stats(self, now_ts: Optional[float] = None) -> Dict[str, Any]:
        now = now_ts if now_ts is not None else time.time()
        with self._lock:
            active = {r["product"]: r["n"] for r in self.conn.execute(
                "SELECT product, COUNT(*) n FROM subscriptions "
                "WHERE status = 'active' AND expires_ts > ? GROUP BY product", (now,))}
            expiring = self.conn.execute(
                "SELECT COUNT(*) n FROM subscriptions WHERE status = 'active' "
                "AND expires_ts BETWEEN ? AND ?", (now, now + 7 * DAY_S)).fetchone()["n"]
            rev = {r["currency"]: r["s"] for r in self.conn.execute(
                "SELECT currency, SUM(amount) s FROM payments "
                "WHERE status = 'paid' AND method != 'admin' AND paid_ts > ? "
                "GROUP BY currency", (now - 30 * DAY_S,))}
        return {"active": active, "expiring_7d": expiring, "revenue_30d": rev}
