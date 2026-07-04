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
from typing import Any, Dict, Optional

import config
from membership.bot import activate_and_welcome, _fmt_date
from membership.payments import LINK_TTL_S, TRON_TTL_S, match_transfers
from membership.plans import SKUS

logger = logging.getLogger("membership.jobs")

POLL_INTERVAL_S = 30
SWEEP_INTERVAL_S = 3600


def _renew_keyboard(product: str):
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup
    sku = "PRO-30" if product == "pro" else "SIG-30"
    return InlineKeyboardMarkup([[
        InlineKeyboardButton(f"Renew 30d — ₹{SKUS[sku].inr}", callback_data=f"sub|{sku}"),
        InlineKeyboardButton(f"Bundle — ₹{SKUS['BUN-30'].inr}", callback_data="sub|BUN-30"),
    ]])


async def poll_payments_once(bd: Dict[str, Any], bot,
                             now_ts: Optional[float] = None) -> int:
    """One pass over pending payments. Returns number of activations."""
    now = now_ts if now_ts is not None else time.time()
    subs = bd["subs"]
    channel_id = bd.get("channel_id")
    activated = 0

    rzp = bd.get("rzp")
    if rzp is not None and rzp.configured:
        for p in subs.pending_payments(method="razorpay"):
            try:
                if p["created_ts"] + LINK_TTL_S + 120 < now:
                    subs.expire_payment(p["id"])
                    continue
                if not p.get("ref"):
                    continue
                status = rzp.link_status(p["ref"])
                if status == "paid":
                    if await activate_and_welcome(bot, subs, p, channel_id,
                                                  ref=p["ref"], now_ts=now):
                        activated += 1
                elif status in ("expired", "cancelled"):
                    subs.expire_payment(p["id"])
            except Exception as e:
                logger.warning("razorpay poll failed for %s: %s", p["id"], e)

    tron = bd.get("tron")
    if tron is not None and tron.configured:
        pendings = subs.pending_payments(method="tron")
        fresh = [p for p in pendings if p["created_ts"] + TRON_TTL_S > now]
        for p in pendings:
            if p["created_ts"] + TRON_TTL_S <= now:
                subs.expire_payment(p["id"])
        if fresh:
            transfers = tron.incoming(min(p["created_ts"] for p in fresh) - 60)
            for p, tx in match_transfers(fresh, transfers):
                try:
                    if await activate_and_welcome(bot, subs, p, channel_id,
                                                  ref=tx, now_ts=now):
                        activated += 1
                except Exception as e:
                    logger.warning("tron activation failed for %s: %s", p["id"], e)
    return activated


async def lifecycle_sweep_once(bd: Dict[str, Any], bot,
                               now_ts: Optional[float] = None) -> Dict[str, int]:
    """Hourly pass: reminders (T-3d / T-1d), kicks past expiry+grace,
    single winback at +7d. Idempotent — the store dedups every send."""
    now = now_ts if now_ts is not None else time.time()
    subs = bd["subs"]
    channel_id = bd.get("channel_id")
    stats = {"reminded": 0, "kicked": 0, "winback": 0}

    for row, stage in subs.due_reminders(now):
        try:
            when = "3 days" if stage == 1 else "1 day"
            await bot.send_message(
                chat_id=row["user_id"],
                text=f"⏳ Your {row['product']} plan ends in {when} "
                     f"({_fmt_date(row['expires_ts'])}).\nRenew now and access "
                     "continues without a gap:",
                reply_markup=_renew_keyboard(row["product"]))
        except Exception as e:
            logger.debug("reminder DM failed for %s: %s", row["user_id"], e)
        # marked even if the DM failed (user blocked the bot) — never spam-retry
        subs.mark_reminded(row["user_id"], row["product"], stage)
        stats["reminded"] += 1

    for row in subs.due_kicks(now):
        if row["product"] == "signals" and channel_id is not None:
            try:
                # Telegram's "remove" idiom: ban + immediate unban, so the
                # member is out but can rejoin the moment they renew.
                await bot.ban_chat_member(chat_id=channel_id, user_id=row["user_id"])
                await bot.unban_chat_member(chat_id=channel_id, user_id=row["user_id"])
            except Exception as e:
                logger.warning("kick failed for %s: %s", row["user_id"], e)
        subs.mark_kicked(row["user_id"], row["product"])
        stats["kicked"] += 1
        try:
            await bot.send_message(
                chat_id=row["user_id"],
                text=f"Your {row['product']} plan has ended — your seat is held. "
                     "Rejoin in one tap:", reply_markup=_renew_keyboard(row["product"]))
        except Exception:
            pass

    for row in subs.due_winbacks(now):
        subs.mark_winback_sent(row["user_id"], row["product"])
        stats["winback"] += 1
        try:
            await bot.send_message(
                chat_id=row["user_id"],
                text="We held your seat 👀 Come back this week and I'll add "
                     "+3 bonus days to any plan.", reply_markup=_renew_keyboard(row["product"]))
        except Exception:
            pass

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
