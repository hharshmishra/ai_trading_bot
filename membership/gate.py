"""Pro gating for the control bot: subscription check + daily fair-use cap.

Applied to Bot B's command handlers ONLY when config.MEMBERSHIP_ENABLED — with
the flag off the handlers are registered bare and behave exactly as before the
membership system existed (flag-off parity is a test).
"""
from __future__ import annotations

import logging

import config

logger = logging.getLogger("membership.gate")


def requires_pro(subs, bot_username: str = ""):
    """Decorator factory: wrap an async PTB handler so it answers only for
    users with an active 'pro' subscription, within the daily query cap."""
    deep_link = f"https://t.me/{bot_username}?start=pro" if bot_username else "the membership bot"

    def deco(handler):
        async def wrapped(update, context):
            uid = update.effective_user.id if update.effective_user else 0
            if not subs.is_active(uid, "pro"):
                await update.message.reply_text(
                    f"🔒 Pro plan required for agent commands.\nSubscribe: {deep_link}")
                return
            if subs.bump_usage(uid) > config.PRO_DAILY_QUERY_CAP:
                await update.message.reply_text(
                    f"⏳ Daily fair-use reached ({config.PRO_DAILY_QUERY_CAP} queries). "
                    "Resets at 00:00 IST.")
                return
            return await handler(update, context)
        wrapped.__name__ = getattr(handler, "__name__", "wrapped")
        return wrapped
    return deco
