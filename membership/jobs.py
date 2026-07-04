"""Membership lifecycle loop: payment polling + reminders + kicks + winbacks.

Split into pure once-functions (injectable now_ts, testable with a FakeBot)
and a thin forever-loop that runs them — same shape as grader_loop. Every
per-member Telegram/network call is individually try/except'd: one blocked DM
or dead API must never stall the sweep (dev-send lesson).
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

import config
from membership.bot import activate_and_welcome, _fmt_date
from membership.payments import LINK_TTL_S, TRON_TTL_S, match_transfers
from membership.plans import SKUS

logger = logging.getLogger("membership.jobs")

POLL_INTERVAL_S = 30
SWEEP_INTERVAL_S = 3600
TRON_GRACE_S = 600      # a USDT transfer confirming this long past TTL still settles


def _renew_keyboard(product: str):
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup
    sku = "PRO-30" if product == "pro" else "SIG-30"
    return InlineKeyboardMarkup([[
        InlineKeyboardButton(f"Renew 30d — ₹{SKUS[sku].inr}", callback_data=f"sub|{sku}"),
        InlineKeyboardButton(f"Bundle — ₹{SKUS['BUN-30'].inr}", callback_data="sub|BUN-30"),
    ]])


async def _notify_order_expired(bot, p: Dict[str, Any]) -> None:
    """Best-effort DM when an unpaid order dies. The buyer pressed Start to
    place the order, so the DM normally lands; failure must never stall the
    poll. TRON late payments stay rescuable for 24h, hence the /paid hint."""
    if p["method"] == "tron":
        text = (f"⌛ Your USDT order ({p['amount']} USDT) expired.\n"
                "Tap /plans for a fresh order — or if you already sent the "
                "USDT, type /paid and I'll find it on-chain.")
    else:
        text = "⌛ Your payment link expired. Tap /plans for a fresh one."
    try:
        await bot.send_message(chat_id=p["user_id"], text=text)
    except Exception as e:
        logger.debug("expiry DM failed for %s: %s", p["user_id"], e)


async def poll_payments_once(bd: Dict[str, Any], bot,
                             now_ts: Optional[float] = None) -> int:
    """One pass over pending payments. Returns number of activations."""
    now = now_ts if now_ts is not None else time.time()
    subs = bd["subs"]
    channel_id = bd.get("channel_id")
    activated = 0

    # Razorpay: check status FIRST, expire only after TTL+grace and only if
    # NOT paid — a payment confirming near the 15-min boundary must never be
    # expired before its status is read (#9). The blocking HTTP call runs in a
    # thread so it can never freeze the shared event loop (#2).
    rzp = bd.get("rzp")
    for p in subs.pending_payments(method="razorpay"):
        try:
            if p.get("ref") and rzp is not None and rzp.configured:
                status = await asyncio.to_thread(rzp.link_status, p["ref"])
                if status == "paid":
                    if await activate_and_welcome(bot, subs, p, channel_id,
                                                  ref=p["ref"], now_ts=now):
                        activated += 1
                    continue
                if status in ("cancelled", "expired"):
                    subs.expire_payment(p["id"])
                    await _notify_order_expired(bot, p)
                    continue
            # expire regardless of .configured, so an unconfigured/removed rail
            # never strands pending rows (#20)
            if p["created_ts"] + LINK_TTL_S + 120 < now:
                subs.expire_payment(p["id"])
                await _notify_order_expired(bot, p)
        except Exception as e:
            logger.warning("razorpay poll failed for %s: %s", p["id"], e)

    # TRON: match across ALL current pendings (incl. those inside the post-TTL
    # grace window) FIRST, then expire only past TTL+grace — a transfer that
    # confirms a little late still settles (#3). One threaded fetch per tick.
    tron = bd.get("tron")
    pendings = subs.pending_payments(method="tron")
    if pendings and tron is not None and tron.configured:
        try:
            since = min(p["created_ts"] for p in pendings) - 60
            transfers = await asyncio.to_thread(tron.incoming, since)
            for p, tx in match_transfers(pendings, transfers):
                if await activate_and_welcome(bot, subs, p, channel_id,
                                              ref=tx, now_ts=now):
                    activated += 1
        except Exception as e:
            logger.warning("tron poll failed: %s", e)
    for p in subs.pending_payments(method="tron"):     # re-query: activated ones gone
        if p["created_ts"] + TRON_TTL_S + TRON_GRACE_S < now:
            subs.expire_payment(p["id"])
            await _notify_order_expired(bot, p)
    return activated


async def lifecycle_sweep_once(bd: Dict[str, Any], bot,
                               now_ts: Optional[float] = None) -> Dict[str, int]:
    """Hourly pass: reminders (T-3d / T-1d), kicks past expiry+grace,
    single winback at +7d. Idempotent — the store dedups every send."""
    now = now_ts if now_ts is not None else time.time()
    subs = bd["subs"]
    channel_id = bd.get("channel_id")
    stats = {"reminded": 0, "kicked": 0, "winback": 0, "removed": 0}

    def _by_user(rows):
        g: Dict[int, List[Dict[str, Any]]] = {}
        for r in rows:
            g.setdefault(r["user_id"], []).append(r)
        return g

    # Reminders: ONE DM per (user, stage) — a bundle buyer holds two product
    # rows but must not get duplicate reminders (#14). Every product row is
    # still stage-marked so it won't re-fire.
    rem_by_user: Dict[int, Dict[int, list]] = {}
    for row, stage in subs.due_reminders(now):
        rem_by_user.setdefault(row["user_id"], {}).setdefault(stage, []).append(row)
    for uid, stages in rem_by_user.items():
        for stage, rows in stages.items():
            when = "3 days" if stage == 1 else "1 day"
            expiry = max(r["expires_ts"] for r in rows)
            try:
                await bot.send_message(
                    chat_id=uid,
                    text=f"⏳ Your subscription ends in {when} ({_fmt_date(expiry)}).\n"
                         "Renew now and access continues without a gap:",
                    reply_markup=_renew_keyboard(rows[0]["product"]))
            except Exception as e:
                logger.debug("reminder DM failed for %s: %s", uid, e)
            for r in rows:
                subs.mark_reminded(uid, r["product"], stage)
            stats["reminded"] += 1

    # Natural expiry: ONE ban + ONE DM per user; mark every product row kicked.
    for uid, rows in _by_user(subs.due_kicks(now)).items():
        if channel_id is not None and any(r["product"] == "signals" for r in rows):
            try:
                await bot.ban_chat_member(chat_id=channel_id, user_id=uid)
                await bot.unban_chat_member(chat_id=channel_id, user_id=uid)
            except Exception as e:
                logger.warning("kick failed for %s: %s", uid, e)
        for r in rows:
            subs.mark_kicked(uid, r["product"])
        stats["kicked"] += 1
        try:
            await bot.send_message(
                chat_id=uid, text="Your subscription has ended — your seat is held. "
                                  "Rejoin in one tap:",
                reply_markup=_renew_keyboard(rows[0]["product"]))
        except Exception:
            pass

    # Admin /revoke removal retry: revoked signals rows whose live ban failed
    # (#13). No DM, no winback — status stays 'revoked'.
    for row in subs.due_channel_removals():
        uid = row["user_id"]
        removed = True
        if channel_id is not None:
            try:
                await bot.ban_chat_member(chat_id=channel_id, user_id=uid)
                await bot.unban_chat_member(chat_id=channel_id, user_id=uid)
            except Exception as e:
                removed = False
                logger.warning("revoke removal retry failed for %s: %s", uid, e)
        if removed:
            subs.mark_channel_removed(uid, row["product"])
            stats["removed"] += 1

    # Winback: ONE DM per user; mark every product row.
    for uid, rows in _by_user(subs.due_winbacks(now)).items():
        for r in rows:
            subs.mark_winback_sent(uid, r["product"])
        stats["winback"] += 1
        try:
            await bot.send_message(
                chat_id=uid, text="We held your seat 👀 Come back this week and I'll "
                                  "add +3 bonus days to any plan.",
                reply_markup=_renew_keyboard(rows[0]["product"]))
        except Exception:
            pass

    g = subs.gc(now)
    if any(g.values()):
        logger.info("membership gc: %s", g)
    return stats


async def membership_loop(app) -> None:
    """Forever-loop over the once-functions (30s payments / hourly lifecycle)."""
    bd = app.bot_data
    logger.info("membership loop started")
    last_sweep = 0.0
    while True:
        try:
            await poll_payments_once(bd, app.bot)
            now = time.time()
            if now - last_sweep >= SWEEP_INTERVAL_S:
                last_sweep = now
                stats = await lifecycle_sweep_once(bd, app.bot)
                if any(stats.values()):
                    logger.info("lifecycle sweep: %s", stats)
            await asyncio.sleep(POLL_INTERVAL_S)
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error("membership loop error: %s", e)
            await asyncio.sleep(POLL_INTERVAL_S)
