"""Bot D — the membership bot. Storefront (/plans, payment flows), doors
(join-request gate, one-time invites) and admin ops (/grant /revoke /subs).

Everything here is a plain PTB handler reading shared objects from
application.bot_data ({"subs", "rzp", "tron", "channel_id"}), so the whole bot
is unit-testable with the FakeBot pattern from tests/test_phase4_runtime.py.
"""
from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime
from typing import Any, Dict, Optional

import config
from membership.plans import SKUS, plans_text
from membership.store import IST, FingerprintExhausted, SubsStore

logger = logging.getLogger("membership.bot")

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
                               now_ts: Optional[float] = None,
                               allow_expired: bool = False) -> bool:
    """mark_paid + door-opening + welcome DM. Idempotent: a payment already
    consumed (double poll) activates nothing and sends nothing. allow_expired
    lets /paid settle a USDT order whose TTL lapsed before it confirmed."""
    rows = subs.mark_paid(payment["id"], ref=ref, now_ts=now_ts, allow_expired=allow_expired)
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
# reply helpers (None-safe / stale-keyboard-safe)
# --------------------------------------------------------------------------- #
async def _reply(update, context, text, **kw):
    """Reply to a command, tolerating update.message=None (edited messages hit
    the default CommandHandler filter with message=None) by falling back to a
    direct DM — every command reaches the bot in a private chat anyway."""
    msg = update.effective_message
    if msg is not None:
        try:
            return await msg.reply_text(text, **kw)
        except Exception:
            pass
    uid = update.effective_user.id if update.effective_user else None
    if uid is not None:
        try:
            return await context.bot.send_message(chat_id=uid, text=text, **kw)
        except Exception as e:
            logger.debug("reply fallback failed: %s", e)


async def _cb_reply(context, q, text, **kw):
    """Reply to a callback via a fresh DM instead of q.message.reply_text: a tap
    on a keyboard older than 48h delivers an InaccessibleMessage with no
    reply_text (AttributeError)."""
    try:
        await context.bot.send_message(chat_id=q.from_user.id, text=text, **kw)
    except Exception as e:
        logger.debug("callback reply failed: %s", e)


async def _mint_signals_invite(bot, channel_id, now_ts=None):
    """A 48h join-request invite for the signals channel (every join is still
    DB-gated by handle_join_request). Raises on failure — caller decides."""
    now = now_ts if now_ts is not None else time.time()
    inv = await bot.create_chat_invite_link(
        chat_id=channel_id, creates_join_request=True,
        expire_date=int(now + 48 * 3600))
    return inv.invite_link


# --------------------------------------------------------------------------- #
# storefront handlers
# --------------------------------------------------------------------------- #
async def cmd_start(update, context) -> None:
    subs: SubsStore = context.application.bot_data["subs"]
    u = update.effective_user
    if u is None:
        return
    subs.touch_user(u.id, u.username)
    payload = context.args[0] if context.args else ""
    if payload.startswith("ref_"):
        if subs.note_referral(u.id, payload[4:]):
            await _reply(update, context, "🎁 Referral registered — you'll both get "
                         f"+{config.REFERRAL_BONUS_DAYS} days on your first plan.")
    # Active signals subscriber pressing Start (or a /grant recipient who just
    # started the bot) gets a fresh working invite on the spot — this is the
    # self-service door that makes /grant onboarding reliable even though a bot
    # cannot DM a user before they have started it.
    channel_id = context.application.bot_data.get("channel_id")
    if channel_id is not None and subs.is_active(u.id, "signals"):
        try:
            link = await _mint_signals_invite(context.bot, channel_id)
            await _reply(update, context,
                         "✅ You have active signals access. Join the channel "
                         f"(tap, you'll be approved automatically):\n{link}")
        except Exception as e:
            logger.error("start invite mint failed for %s: %s", u.id, e)
    await cmd_plans(update, context)


async def cmd_plans(update, context) -> None:
    await _reply(update, context, plans_text(), parse_mode="HTML",
                 reply_markup=_plans_keyboard(), disable_web_page_preview=True)


async def cmd_ref(update, context) -> None:
    if update.effective_user is None:
        return
    subs: SubsStore = context.application.bot_data["subs"]
    user = subs.touch_user(update.effective_user.id, update.effective_user.username)
    me = await context.bot.get_me()
    await _reply(update, context,
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
        await _cb_reply(context, q,
            f"<b>{s.label}</b> — ₹{s.inr} or ~{s.usdt:g} USDT.\nPick a payment method:",
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
                # threaded: create_link is a blocking requests.post — must not
                # run on the shared event loop that broadcasts signals
                link_id, url = await asyncio.to_thread(
                    rzp.create_link, SKUS[code].inr, SKUS[code].label, uid, code)
            except Exception as e:
                logger.error("razorpay link failed: %s", e)
                subs.expire_payment(p["id"])
                await q.answer("UPI link failed — please try USDT.", show_alert=True)
                return
            subs.set_payment_ref(p["id"], link_id)
            await q.answer()
            await _cb_reply(context, q,
                f"💳 Pay ₹{SKUS[code].inr} here (link valid 15 min):\n{url}\n\n"
                "Access activates automatically within a minute of payment.")
        elif rail == "usdt":
            tron = context.application.bot_data.get("tron")
            if tron is None or not tron.configured:
                await q.answer("USDT temporarily unavailable — please pay via UPI.",
                               show_alert=True)
                return
            try:
                p = subs.create_pending_payment(uid, code, "USDT", "tron")
            except FingerprintExhausted:
                await q.answer("High demand right now — try again in a minute.",
                               show_alert=True)
                return
            await q.answer()
            await _cb_reply(context, q,
                f"₮ Send <b>exactly {p['amount']:.3f} USDT</b> (TRC-20) to:\n"
                f"<code>{tron.wallet}</code>\n\n"
                "A few thousandths are added so we can identify your payment — "
                "send the exact amount shown, do not round it.\n"
                "Access activates automatically after confirmation (~1–2 min).\n"
                "Sent but nothing happened after 5 min? Reply /paid",
                parse_mode="HTML")
        return

    await q.answer()   # unrecognised data — ack so the client spinner stops


async def cmd_paid(update, context) -> None:
    """Manual USDT nudge: force an immediate on-chain check for this user's
    order — INCLUDING one whose TTL just lapsed before its transfer confirmed
    (rescuable_tron_payments spans expired-within-24h). Threaded HTTP."""
    if update.effective_user is None:
        return
    subs: SubsStore = context.application.bot_data["subs"]
    tron = context.application.bot_data.get("tron")
    uid = update.effective_user.id
    mine = subs.rescuable_tron_payments(uid)
    if not mine or tron is None or not tron.configured:
        await _reply(update, context, "No recent USDT order found — /plans to start one.")
        return
    from membership.payments import match_transfers
    transfers = await asyncio.to_thread(tron.incoming, min(p["created_ts"] for p in mine) - 60)
    matched = match_transfers(mine, transfers)
    if not matched:
        await _reply(update, context,
            "Nothing matching on-chain yet — TRON confirmations take ~1–2 min. "
            "I'll keep checking automatically.")
        return
    channel_id = context.application.bot_data.get("channel_id")
    for p, tx in matched:
        await activate_and_welcome(context.bot, subs, p, channel_id, ref=tx,
                                   allow_expired=True)


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
    if len(context.args) < 3 or context.args[2] not in ("signals", "pro"):
        await _reply(update, context, "Usage: /grant <user_id> <days> <signals|pro>")
        return
    try:
        uid, days, product = int(context.args[0]), int(context.args[1]), context.args[2]
    except ValueError:
        await _reply(update, context, "user_id and days must be numbers.")
        return
    subs: SubsStore = context.application.bot_data["subs"]
    row = subs.grant(uid, days, product)
    note = f"Granted {product} to {uid} until {_fmt_date(row['expires_ts'])}."

    if product == "signals":
        channel_id = context.application.bot_data.get("channel_id")
        link = None
        if channel_id is not None:
            try:
                link = await _mint_signals_invite(context.bot, channel_id)
            except Exception as e:
                logger.error("grant invite mint failed: %s", e)
                note += (f"\n⚠️ Could not create the invite link: {e}. "
                         "Check the bot has the 'Invite Users via Link' admin right.")
        if link:
            # A bot CANNOT DM a user who has never pressed Start on it, so this
            # send often fails for a brand-new grantee — surface that to the
            # admin with the link + instructions rather than reporting a bare
            # success the grantee never received.
            try:
                await context.bot.send_message(
                    chat_id=uid, text=f"🎟 You've been granted {days} days of signals "
                                      f"access.\nJoin: {link}")
                note += "\n✅ Invite DM'd to the user."
            except Exception:
                note += ("\n⚠️ Could not DM the user (they must press Start on this "
                         "bot first). Forward them this invite, or ask them to open "
                         f"the bot and press Start:\n{link}")
    else:  # pro
        try:
            await context.bot.send_message(
                chat_id=uid, text=f"🎟 You've been granted {days} days of Pro access — "
                                  "DM the control bot /news /indicator /regime …")
            note += "\n✅ Notified the user."
        except Exception:
            note += "\n⚠️ Could not DM the user (they must press Start on the bot first)."
    await _reply(update, context, note)


async def cmd_revoke(update, context) -> None:
    if not _is_admin(update):
        return
    if len(context.args) < 2 or context.args[1] not in ("signals", "pro"):
        await _reply(update, context, "Usage: /revoke <user_id> <signals|pro>")
        return
    try:
        uid, product = int(context.args[0]), context.args[1]
    except ValueError:
        await _reply(update, context, "user_id must be a number.")
        return
    subs: SubsStore = context.application.bot_data["subs"]
    ok = subs.revoke(uid, product)
    # Immediate best-effort removal; if the ban fails the row stays
    # channel_removed=0 and the hourly sweep (due_channel_removals) retries.
    note = "Revoked."
    if ok and product == "signals":
        channel_id = context.application.bot_data.get("channel_id")
        if channel_id is not None:
            try:
                await context.bot.ban_chat_member(chat_id=channel_id, user_id=uid)
                await context.bot.unban_chat_member(chat_id=channel_id, user_id=uid)
                subs.mark_channel_removed(uid, product)
                note = "Revoked + removed from channel."
            except Exception as e:
                logger.error("revoke kick failed for %s: %s", uid, e)
                note = "Revoked (access off). Channel removal will retry on the next sweep."
    await _reply(update, context, note if ok else "No such subscription.")


async def cmd_subs(update, context) -> None:
    if not _is_admin(update):
        return
    subs: SubsStore = context.application.bot_data["subs"]
    s = subs.stats()
    rev = " · ".join(f"{v:,.0f} {k}" for k, v in (s["revenue_30d"] or {}).items()) or "0"
    await _reply(update, context,
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
