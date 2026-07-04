"""Bot D — the membership bot. Storefront (/plans, payment flows), doors
(join-request gate, one-time invites) and admin ops (/grant /revoke /subs).

Everything here is a plain PTB handler reading shared objects from
application.bot_data ({"subs", "rzp", "tron", "channel_id"}), so the whole bot
is unit-testable with the FakeBot pattern from tests/test_phase4_runtime.py.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

import config
from membership.plans import SKUS, plans_text
from membership.store import SubsStore

logger = logging.getLogger("membership.bot")
IST = timezone(timedelta(hours=5, minutes=30))

DISCLAIMER = "\n<i>Educational market analysis, not financial advice.</i>"


def _fmt_date(ts: float) -> str:
    return datetime.fromtimestamp(ts, IST).strftime("%d %b %Y, %H:%M IST")


def _plans_keyboard():
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup
    rows, row = [], []
    for code in ("SIG-7", "SIG-15", "SIG-30", "PRO-15", "PRO-30", "BUN-30", "FND-90"):
        s = SKUS[code]
        row.append(InlineKeyboardButton(f"{s.label} · ₹{s.inr}", callback_data=f"sub|{code}"))
        if len(row) == 2:
            rows.append(row)
            row = []
    if row:
        rows.append(row)
    return InlineKeyboardMarkup(rows)


def _rail_keyboard(code: str):
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup
    s = SKUS[code]
    return InlineKeyboardMarkup([[
        InlineKeyboardButton(f"🇮🇳 UPI · ₹{s.inr}", callback_data=f"pay|{code}|inr"),
        InlineKeyboardButton(f"₮ USDT · {s.usdt:g}", callback_data=f"pay|{code}|usdt"),
    ]])


async def activate_and_welcome(bot, subs: SubsStore, payment: Dict[str, Any],
                               channel_id: Optional[int], ref: Optional[str] = None,
                               now_ts: Optional[float] = None) -> bool:
    """mark_paid + door-opening + welcome DM. Idempotent: a payment already
    consumed (double poll) activates nothing and sends nothing."""
    rows = subs.mark_paid(payment["id"], ref=ref, now_ts=now_ts)
    if not rows:
        return False
    sku = SKUS[payment["sku"]]
    uid = payment["user_id"]
    user = subs.touch_user(uid, now_ts=now_ts)
    expiry = max(r["expires_ts"] for r in rows)

    lines = [f"✅ You're in! <b>{sku.label}</b> active until <b>{_fmt_date(expiry)}</b>."]
    if "signals" in sku.products and channel_id is not None:
        try:
            now = now_ts if now_ts is not None else time.time()
            # creates_join_request (NOT member_limit): a member_limit link
            # admits whoever taps it first WITHOUT the database check — an
            # unused link would let a revoked/lapsed user (or anyone they
            # forwarded it to) bypass the gate for 48h. Join-request links
            # route EVERY join through handle_join_request, which approves
            # strictly by subscription state, so a stale link is harmless.
            invite = await bot.create_chat_invite_link(
                chat_id=channel_id, creates_join_request=True,
                expire_date=int(now + 48 * 3600))
            lines.append("Your invite (valid 48h — tap it and you'll be "
                         f"approved automatically):\n{invite.invite_link}")
        except Exception as e:
            logger.error("invite mint failed for %s: %s", uid, e)
            lines.append("Invite link pending — use /start if it doesn't arrive shortly.")
        lines.append("Signals arrive at :30 IST — 1h hourly · 4h ×6/day · daily 05:30 · weekly Mon 05:30.")
    if "pro" in sku.products:
        lines.append("Pro commands are live: /news /indicator /research /context /regime /derivs "
                     "— DM them to the control bot.")
    lines.append(f"Your referral code: <b>{user['referral_code']}</b> — every friend who "
                 f"joins gives you both +{config.REFERRAL_BONUS_DAYS} days.")
    try:
        await bot.send_message(chat_id=uid, text="\n".join(lines) + DISCLAIMER,
                               parse_mode="HTML")
    except Exception as e:
        logger.error("welcome DM failed for %s: %s", uid, e)
    return True


# --------------------------------------------------------------------------- #
# storefront handlers
# --------------------------------------------------------------------------- #
async def cmd_start(update, context) -> None:
    subs: SubsStore = context.application.bot_data["subs"]
    u = update.effective_user
    subs.touch_user(u.id, u.username)
    payload = context.args[0] if context.args else ""
    if payload.startswith("ref_"):
        if subs.note_referral(u.id, payload[4:]):
            await update.message.reply_text("🎁 Referral registered — you'll both get "
                                            f"+{config.REFERRAL_BONUS_DAYS} days on your first plan.")
    await cmd_plans(update, context)


async def cmd_plans(update, context) -> None:
    await update.message.reply_text(plans_text(), parse_mode="HTML",
                                    reply_markup=_plans_keyboard(),
                                    disable_web_page_preview=True)


async def cmd_ref(update, context) -> None:
    subs: SubsStore = context.application.bot_data["subs"]
    user = subs.touch_user(update.effective_user.id, update.effective_user.username)
    me = await context.bot.get_me()
    await update.message.reply_text(
        f"Your referral code: {user['referral_code']}\n"
        f"Share: https://t.me/{me.username}?start=ref_{user['referral_code']}\n"
        f"Every friend's first payment gives you BOTH +{config.REFERRAL_BONUS_DAYS} days.")


async def handle_membership_callback(update, context) -> None:
    q = update.callback_query
    if not q or "|" not in (q.data or ""):
        return
    subs: SubsStore = context.application.bot_data["subs"]
    parts = q.data.split("|")

    if parts[0] == "sub" and len(parts) == 2 and parts[1] in SKUS:
        code = parts[1]
        s = SKUS[code]
        await q.answer()
        await q.message.reply_text(
            f"<b>{s.label}</b> — ₹{s.inr} or {s.usdt:g} USDT.\nPick a payment method:",
            parse_mode="HTML", reply_markup=_rail_keyboard(code))
        return

    if parts[0] == "pay" and len(parts) == 3 and parts[1] in SKUS:
        code, rail = parts[1], parts[2]
        uid = q.from_user.id
        subs.touch_user(uid, q.from_user.username)
        if rail == "inr":
            rzp = context.application.bot_data.get("rzp")
            if rzp is None or not rzp.configured:
                await q.answer("UPI temporarily unavailable — please pay via USDT.",
                               show_alert=True)
                return
            p = subs.create_pending_payment(uid, code, "INR", "razorpay")
            try:
                link_id, url = rzp.create_link(SKUS[code].inr, SKUS[code].label, uid, code)
            except Exception as e:
                logger.error("razorpay link failed: %s", e)
                subs.expire_payment(p["id"])
                await q.answer("UPI link failed — please try USDT.", show_alert=True)
                return
            subs.set_payment_ref(p["id"], link_id)
            await q.answer()
            await q.message.reply_text(
                f"💳 Pay ₹{SKUS[code].inr} here (link valid 15 min):\n{url}\n\n"
                "Access activates automatically within a minute of payment.")
        elif rail == "usdt":
            tron = context.application.bot_data.get("tron")
            if tron is None or not tron.configured:
                await q.answer("USDT temporarily unavailable — please pay via UPI.",
                               show_alert=True)
                return
            p = subs.create_pending_payment(uid, code, "USDT", "tron")
            await q.answer()
            await q.message.reply_text(
                f"₮ Send <b>exactly {p['amount']:.3f} USDT</b> (TRC-20) to:\n"
                f"<code>{tron.wallet}</code>\n\n"
                "The exact amount is how we match your payment — do not round it.\n"
                "Access activates automatically after confirmation (~1–2 min).\n"
                "Sent but nothing happened after 5 min? Reply /paid",
                parse_mode="HTML")
        return


async def cmd_paid(update, context) -> None:
    """Manual USDT nudge: force an immediate on-chain sweep for this user's
    pending order instead of waiting for the next poll tick."""
    subs: SubsStore = context.application.bot_data["subs"]
    tron = context.application.bot_data.get("tron")
    uid = update.effective_user.id
    mine = [p for p in subs.pending_payments(method="tron") if p["user_id"] == uid]
    if not mine or tron is None:
        await update.message.reply_text("No pending USDT order found — /plans to start one.")
        return
    from membership.payments import match_transfers
    transfers = tron.incoming(min(p["created_ts"] for p in mine) - 60)
    matched = match_transfers(mine, transfers)
    if not matched:
        await update.message.reply_text(
            "Nothing matching on-chain yet — TRON confirmations take ~1–2 min. "
            "I'll keep checking automatically.")
        return
    channel_id = context.application.bot_data.get("channel_id")
    for p, tx in matched:
        await activate_and_welcome(context.bot, subs, p, channel_id, ref=tx)


# --------------------------------------------------------------------------- #
# doors
# --------------------------------------------------------------------------- #
async def handle_join_request(update, context) -> None:
    """The real gate on the signals channel: approve join requests only for
    active subscribers — a leaked invite link is worthless."""
    req = update.chat_join_request
    if req is None:
        return
    subs: SubsStore = context.application.bot_data["subs"]
    uid = req.from_user.id
    if subs.is_active(uid, "signals"):
        await req.approve()
        return
    await req.decline()
    try:
        me = await context.bot.get_me()
        await context.bot.send_message(
            chat_id=uid, text="This channel is for subscribers. Plans: "
                              f"https://t.me/{me.username}?start=plans")
    except Exception:
        pass                # user never talked to the bot -> DM impossible; fine


# --------------------------------------------------------------------------- #
# admin
# --------------------------------------------------------------------------- #
def _is_admin(update) -> bool:
    return bool(update.effective_user and update.effective_user.id in config.ADMIN_USER_IDS)


async def cmd_grant(update, context) -> None:
    if not _is_admin(update):
        return
    try:
        uid, days, product = int(context.args[0]), int(context.args[1]), context.args[2]
        assert product in ("signals", "pro")
    except Exception:
        await update.message.reply_text("Usage: /grant <user_id> <days> <signals|pro>")
        return
    subs: SubsStore = context.application.bot_data["subs"]
    row = subs.grant(uid, days, product)
    if product == "signals":
        channel_id = context.application.bot_data.get("channel_id")
        if channel_id is not None:
            try:
                invite = await context.bot.create_chat_invite_link(
                    chat_id=channel_id, creates_join_request=True,
                    expire_date=int(time.time() + 48 * 3600))
                await context.bot.send_message(
                    chat_id=uid, text=f"🎟 You've been granted {days} days of signals "
                                      f"access.\nJoin: {invite.invite_link}")
            except Exception as e:
                logger.error("grant invite failed: %s", e)
    await update.message.reply_text(
        f"Granted {product} to {uid} until {_fmt_date(row['expires_ts'])}")


async def cmd_revoke(update, context) -> None:
    if not _is_admin(update):
        return
    try:
        uid, product = int(context.args[0]), context.args[1]
    except Exception:
        await update.message.reply_text("Usage: /revoke <user_id> <signals|pro>")
        return
    subs: SubsStore = context.application.bot_data["subs"]
    ok = subs.revoke(uid, product)
    # the hourly kicker only sweeps status='active' rows (natural expiry), so
    # a revoke must remove channel access RIGHT HERE or the member stays in
    if ok and product == "signals":
        channel_id = context.application.bot_data.get("channel_id")
        if channel_id is not None:
            try:
                await context.bot.ban_chat_member(chat_id=channel_id, user_id=uid)
                await context.bot.unban_chat_member(chat_id=channel_id, user_id=uid)
            except Exception as e:
                logger.error("revoke kick failed for %s: %s", uid, e)
    await update.message.reply_text("Revoked + removed." if ok else "No such subscription.")


async def cmd_subs(update, context) -> None:
    if not _is_admin(update):
        return
    subs: SubsStore = context.application.bot_data["subs"]
    s = subs.stats()
    rev = " · ".join(f"{v:,.0f} {k}" for k, v in (s["revenue_30d"] or {}).items()) or "0"
    await update.message.reply_text(
        f"📊 active: {s['active'] or {}}\nexpiring ≤7d: {s['expiring_7d']}\n"
        f"revenue 30d: {rev}")


def register(app, subs: SubsStore, channel_id: Optional[int],
             rzp=None, tron=None) -> None:
    """Attach the full handler set + shared objects to a PTB Application."""
    from telegram.ext import (CallbackQueryHandler, ChatJoinRequestHandler,
                              CommandHandler)
    app.bot_data.update({"subs": subs, "channel_id": channel_id,
                         "rzp": rzp, "tron": tron})
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("plans", cmd_plans))
    app.add_handler(CommandHandler("ref", cmd_ref))
    app.add_handler(CommandHandler("paid", cmd_paid))
    app.add_handler(CommandHandler("grant", cmd_grant))
    app.add_handler(CommandHandler("revoke", cmd_revoke))
    app.add_handler(CommandHandler("subs", cmd_subs))
    app.add_handler(CallbackQueryHandler(handle_membership_callback))
    app.add_handler(ChatJoinRequestHandler(handle_join_request))
