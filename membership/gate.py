"""Pro gating for the control bot: subscription check + daily fair-use cap.

Applied to Bot B's command handlers ONLY when config.MEMBERSHIP_ENABLED — with
the flag off the handlers are registered bare and behave exactly as before the
membership system existed (flag-off parity is a test).
"""
from __future__ import annotations

import logging

import config

logger = logging.getLogger("membership.gate")


def requires_pro(subs):
    """Decorator factory: wrap an async PTB handler so it answers only for
    users with an active 'pro' subscription, within the daily query cap.

    The subscribe deep link is resolved at CALL time from
    bot_data['membership_username'] (set in post_init after the storefront
    bot's get_me()), so the refusal always carries a tappable t.me link — the
    old build-time bot_username arg was never supplied at the wiring site and
    rendered as literal filler."""
    def deco(handler):
        async def wrapped(update, context):
            u = update.effective_user
            if u is None:
                return                       # service/anonymous update — nothing to gate
            uid = u.id
            msg = update.effective_message

            async def _say(text):
                if msg is not None:
                    try:
                        await msg.reply_text(text)
                        return
                    except Exception:
                        pass                 # edited/inaccessible message → DM fallback
                try:
                    await context.bot.send_message(chat_id=uid, text=text)
                except Exception as e:
                    logger.debug("gate reply failed for %s: %s", uid, e)

            if not subs.is_active(uid, "pro"):
                username = (context.application.bot_data or {}).get("membership_username")
                link = f"https://t.me/{username}?start=pro" if username else "the membership bot"
                await _say(f"🔒 Pro plan required for agent commands.\nSubscribe: {link}")
                return
            if subs.bump_usage(uid) > config.PRO_DAILY_QUERY_CAP:
                await _say(f"⏳ Daily fair-use reached ({config.PRO_DAILY_QUERY_CAP} queries). "
                           "Resets at 00:00 IST.")
                return
            return await handler(update, context)
        wrapped.__name__ = getattr(handler, "__name__", "wrapped")
        return wrapped
    return deco
