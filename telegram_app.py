"""Telegram runtime (Phase 4).

One long-lived python-telegram-bot Application with the scheduler and the grader
running as background asyncio tasks in the SAME loop (via post_init) — this is
what makes inline-button feedback reliable: there is always a running app to
receive the callback, and the reward is applied from the durable SQLite record
(not clobbered instance state).

Bots/channels:
  * Bot A (signals): customer channel = signal only; dev channel = signal +
    brain dump + 3-row feedback keyboard.
  * Bot B (control, optional separate token): /news /indicator /research /context
    to invoke any child agent on demand.

The pure send/session logic lives in Broadcaster and handle_callback so it is
unit-testable with a fake bot; the wiring in main() needs live tokens.
"""
from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timedelta
from typing import Any, Dict, Optional
from zoneinfo import ZoneInfo

from grader import Grader
from persistence import Store, get_store
from signals import (TF_SECONDS, fmt_brain_dump, fmt_signal_message,
                     is_candle_close_minute, timeframes_due)
from cycle import run_cycle

IST = ZoneInfo("Asia/Kolkata")
logger = logging.getLogger("bitreinforcex.runtime")

SESSION_TTL_HOURS = int(os.getenv("SESSION_TTL_HOURS", "12"))
GRADER_INTERVAL_S = int(os.getenv("GRADER_INTERVAL_S", "60"))
MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY", "5"))


# --------------------------------------------------------------------------- #
# Inline keyboard
# --------------------------------------------------------------------------- #
def build_dev_keyboard(session_id: str):
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("BUY", callback_data=f"{session_id}|OUTCOME|buy"),
         InlineKeyboardButton("SELL", callback_data=f"{session_id}|OUTCOME|sell"),
         InlineKeyboardButton("SKIP LEARNING", callback_data=f"{session_id}|OUTCOME|skip")],
        [InlineKeyboardButton("1.0", callback_data=f"{session_id}|REWARD|1.0"),
         InlineKeyboardButton("-4.0", callback_data=f"{session_id}|REWARD|-4.0"),
         InlineKeyboardButton("Auto-Assign", callback_data=f"{session_id}|REWARD|auto")],
        [InlineKeyboardButton("CLOSE SESSION", callback_data=f"{session_id}|CLOSE|x")],
    ])


# --------------------------------------------------------------------------- #
# Broadcaster — sends signals + manages durable sessions
# --------------------------------------------------------------------------- #
class Broadcaster:
    def __init__(self, bot, store: Store, customer_chat_id: Optional[int],
                 dev_chat_id: Optional[int]):
        self.bot = bot
        self.store = store
        self.customer_chat_id = customer_chat_id
        self.dev_chat_id = dev_chat_id

    async def broadcast(self, *, pair: str, tf: str, overall: str, nwe: str,
                        conf: float, reason: str, decision: Dict[str, Any]) -> Optional[str]:
        """Send to customer + dev channels, create a durable session, supersede
        the previous one for this (pair, tf). Returns the new session id."""
        from telegram.constants import ParseMode
        text = fmt_signal_message(pair, tf, overall, nwe, conf, reason)

        cust_msg_id = None
        if self.customer_chat_id is not None:
            try:
                m = await self.bot.send_message(chat_id=self.customer_chat_id, text=text,
                                                parse_mode=ParseMode.HTML, disable_web_page_preview=True)
                cust_msg_id = m.message_id
            except Exception as e:
                logger.error("customer send failed: %s", e)

        # Create the session first so its id is on the dev keyboard.
        session_id = self.store.create_session(
            pair=pair, tf=tf, customer_chat_id=self.customer_chat_id, customer_msg_id=cust_msg_id,
            dev_chat_id=self.dev_chat_id)

        dev_msg_id = None
        if self.dev_chat_id is not None:
            try:
                dump = text + "\n\n" + fmt_brain_dump(decision)
                m = await self.bot.send_message(chat_id=self.dev_chat_id, text=dump,
                                                parse_mode=ParseMode.HTML, disable_web_page_preview=True,
                                                reply_markup=build_dev_keyboard(session_id))
                dev_msg_id = m.message_id
            except Exception as e:
                logger.error("dev send failed: %s", e)

        # Persist message ids + supersede the previous active session.
        with self.store._lock:
            self.store.conn.execute("UPDATE sessions SET dev_msg_id = ? WHERE id = ?",
                                    (dev_msg_id, session_id))
            self.store.conn.commit()
        self.store.supersede_active(pair, tf, session_id)
        return session_id

    async def strip_keyboard(self, session: Dict[str, Any]) -> None:
        if session and session.get("dev_chat_id") and session.get("dev_msg_id"):
            try:
                await self.bot.edit_message_reply_markup(
                    chat_id=session["dev_chat_id"], message_id=session["dev_msg_id"], reply_markup=None)
            except Exception as e:
                logger.debug("strip keyboard failed: %s", e)


# --------------------------------------------------------------------------- #
# Callback handler — routes button presses to the grader (durable, race-free)
# --------------------------------------------------------------------------- #
async def handle_callback(update, context) -> None:
    q = update.callback_query
    if not q:
        return
    store: Store = context.application.bot_data["store"]
    grader: Grader = context.application.bot_data["grader"]
    broadcaster: Broadcaster = context.application.bot_data["broadcaster"]

    try:
        session_id, kind, value = q.data.split("|", 2)
    except Exception:
        await q.answer("Malformed callback.", show_alert=True)
        return

    sess = store.get_session(session_id)
    if not sess or not sess.get("active"):
        await q.answer("Session is not active.")
        return

    if kind == "CLOSE":
        store.deactivate_session(session_id)
        await broadcaster.strip_keyboard(sess)
        await q.answer("Session closed.")
        return

    if kind == "OUTCOME":
        store.set_session_true_outcome(session_id, value)
        await q.answer(f"Outcome set: {value.upper()}")
        return

    if kind == "REWARD":
        true_outcome = (store.get_session(session_id) or {}).get("true_outcome")
        if not true_outcome:
            await q.answer("Select BUY/SELL/SKIP first.", show_alert=True)
            return
        if true_outcome == "skip":
            store.deactivate_session(session_id)
            await broadcaster.strip_keyboard(sess)
            await q.answer("Learning skipped. Session closed.")
            return
        news_reward = None if value == "auto" else float(value)
        result = grader.apply_manual_feedback(sess["prediction_id"], true_outcome, news_reward=news_reward)
        store.deactivate_session(session_id)
        await broadcaster.strip_keyboard(sess)
        await q.answer(f"Feedback applied ({result.get('status')}). Session closed.")


# --------------------------------------------------------------------------- #
# Control bot commands
# --------------------------------------------------------------------------- #
async def _reply(update, text: str):
    from telegram.constants import ParseMode
    await update.message.reply_text(text[:4000], parse_mode=ParseMode.HTML)


async def cmd_news(update, context):
    dm = context.application.bot_data["dm"]
    pair = (context.args[0] if context.args else "BTCUSDT").upper()
    out = await asyncio.to_thread(dm.news.run, pair)
    await _reply(update, f"<b>news {pair}</b>\naction={out.get('action')} conf={out.get('confidence')}\n"
                         f"overall={out.get('overall_json', {}).get('sentiment')} "
                         f"pair={out.get('pair_json', {}).get('sentiment')}")


async def cmd_indicator(update, context):
    dm = context.application.bot_data["dm"]
    pair = (context.args[0] if context.args else "BTCUSDT").upper()
    tf = context.args[1] if len(context.args) > 1 else "4h"
    out = await asyncio.to_thread(dm.indicator.decide, pair, tf)
    await _reply(update, f"<b>indicator {pair} {tf}</b>\naction={out.action} conf={round(out.confidence,3)}\n"
                         f"type1={out.details['type1']['action']} type2={out.details['type2']['action']}")


async def cmd_research(update, context):
    dm = context.application.bot_data["dm"]
    pair = (context.args[0] if context.args else "ETHUSDT").upper()
    tf = context.args[1] if len(context.args) > 1 else "4h"
    out = await asyncio.to_thread(dm.research.decide, pair, tf, dm.indicator, dm.news)
    await _reply(update, f"<b>research {pair} {tf}</b>\naction={out.get('action')} conf={out.get('confidence')}")


async def cmd_regime(update, context):
    pair = (context.args[0] if context.args else "BTCUSDT").upper()
    tf = context.args[1] if len(context.args) > 1 else "4h"
    from agents.regime_agent import RegimeAgent
    out = await asyncio.to_thread(RegimeAgent().decide, pair, tf)
    await _reply(update, f"<b>regime {pair} {tf}</b>\nregime={out.get('regime')}\n"
                         f"adx={_r3(out.get('adx'))} chop={_r3(out.get('chop'))} "
                         f"vol_pct={_r3(out.get('vol_pct'))} atr={_r3(out.get('atr'))}")


def _r3(v):
    return round(v, 3) if isinstance(v, (int, float)) else v


async def cmd_context(update, context):
    dm = context.application.bot_data["dm"]
    tf = context.args[0] if context.args else "4h"
    from market_context import build_market_context
    from cycle import SYMBOLS
    ctx = await asyncio.to_thread(build_market_context, tf, SYMBOLS, dm.indicator, dm.news, dm.research)
    await _reply(update, f"<b>market context {tf}</b>\nspx={round(ctx.spx_score,3)} dxy={round(ctx.dxy_score,3)} "
                         f"money_flow={round(ctx.money_flow_phase,3)} btc_dom={round(ctx.btdom_effect,3)}\n"
                         f"drivers={len(ctx.driver_ind_score)}")


# --------------------------------------------------------------------------- #
# Background loops
# --------------------------------------------------------------------------- #
async def scheduler_loop(application) -> None:
    bd = application.bot_data
    last_run = None
    logger.info("scheduler started")
    while True:
        try:
            dt = datetime.now(tz=IST).replace(second=0, microsecond=0)
            if is_candle_close_minute(dt) and dt != last_run:
                last_run = dt
                tfs = timeframes_due(dt)
                logger.info("cycle at %s for %s", dt.strftime("%Y-%m-%d %H:%M"), tfs)
                await run_cycle(tfs, dm=bd["dm"], data_fetcher=bd["dm"].indicator.data,
                                broadcast=bd["broadcaster"].broadcast, store=bd["store"],
                                concurrency=MAX_CONCURRENCY)
                store: Store = bd["store"]
                stale = store.gc_sessions((datetime.now(tz=IST) - timedelta(hours=SESSION_TTL_HOURS)).timestamp())
                for s in stale:
                    await bd["broadcaster"].strip_keyboard(s)
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error("scheduler error: %s", e)
            await asyncio.sleep(60)


async def grader_loop(application) -> None:
    bd = application.bot_data
    logger.info("grader started (every %ss)", GRADER_INTERVAL_S)
    while True:
        try:
            graded = await asyncio.to_thread(bd["grader"].grade_once)
            if graded:
                logger.info("auto-graded %d predictions", len(graded))
            await asyncio.sleep(GRADER_INTERVAL_S)
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error("grader error: %s", e)
            await asyncio.sleep(GRADER_INTERVAL_S)


# --------------------------------------------------------------------------- #
# App wiring
# --------------------------------------------------------------------------- #
def _chat_id(*names: str) -> Optional[int]:
    for n in names:
        v = os.getenv(n)
        if v:
            try:
                return int(v)
            except ValueError:
                return None
    return None


def main() -> None:
    from dotenv import load_dotenv
    load_dotenv()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    from telegram import Update
    from telegram.ext import Application, CallbackQueryHandler, CommandHandler

    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        logger.error("TELEGRAM_BOT_TOKEN not set")
        return

    from brain.decision_maker import DecisionMaker
    dm = DecisionMaker(prefer_csv=False)
    store = get_store()
    grader = Grader(dm, data_fetcher=dm.indicator.data, store=store)

    customer = _chat_id("CUSTOMER_CHAT_ID", "TELEGRAM_SIGNALS_CHANNEL_ID")
    dev = _chat_id("DEV_CHAT_ID", "TELEGRAM_DEV_CHANNEL_ID")

    control_token = os.environ.get("TELEGRAM_CONTROL_BOT_TOKEN")
    control_app = None

    async def post_init(app):
        broadcaster = Broadcaster(app.bot, store, customer, dev)
        app.bot_data.update({"dm": dm, "store": store, "grader": grader, "broadcaster": broadcaster})
        app.bot_data["_tasks"] = [asyncio.create_task(scheduler_loop(app)),
                                  asyncio.create_task(grader_loop(app))]
        if control_app is not None:
            control_app.bot_data["dm"] = dm
            await control_app.initialize()
            await control_app.start()
            await control_app.updater.start_polling(allowed_updates=Update.ALL_TYPES)
            logger.info("control bot started")

    async def post_shutdown(app):
        for t in app.bot_data.get("_tasks", []):
            t.cancel()
        if control_app is not None:
            await control_app.updater.stop()
            await control_app.stop()
            await control_app.shutdown()

    app = Application.builder().token(token).post_init(post_init).post_shutdown(post_shutdown).build()
    app.add_handler(CallbackQueryHandler(handle_callback))

    if control_token:
        control_app = Application.builder().token(control_token).build()
        control_app.add_handler(CommandHandler("news", cmd_news))
        control_app.add_handler(CommandHandler("indicator", cmd_indicator))
        control_app.add_handler(CommandHandler("research", cmd_research))
        control_app.add_handler(CommandHandler("context", cmd_context))
        control_app.add_handler(CommandHandler("regime", cmd_regime))

    logger.info("starting BitReinforceX runtime")
    app.run_polling(poll_interval=1.0, allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
