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
from ingestion import ingest_all
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
    """One-tap verdict keyboard (v3.2): a single press records what ACTUALLY
    happened and trains every agent + the brain against it with the active
    reward map. FLAT is a real, teachable verdict (skip-callers rewarded,
    directional calls get the timeout penalty) — previously impossible. The
    old two-step OUTCOME→REWARD keyboard (with the news-only 1.0/−4.0
    overrides, a v1 relic) is gone from NEW messages, but its callbacks are
    still handled so old messages in the channel keep working."""
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("✅ BUY", callback_data=f"{session_id}|VERDICT|buy"),
         InlineKeyboardButton("❌ SELL", callback_data=f"{session_id}|VERDICT|sell"),
         InlineKeyboardButton("➖ FLAT", callback_data=f"{session_id}|VERDICT|skip")],
        [InlineKeyboardButton("🚫 DISCARD (no learning)", callback_data=f"{session_id}|CLOSE|x")],
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
        ind_details = (((decision.get("agents") or {}).get("indicator") or {})
                       .get("raw") or {}).get("details") or {}
        deriv_raw = ((decision.get("agents") or {}).get("derivatives") or {}).get("raw") or {}
        note = None
        try:
            from agents.derivatives_agent import deriv_note as _dn
            note = _dn(deriv_raw.get("details")) if deriv_raw.get("available") else None
        except Exception:
            note = None
        meta = decision.get("meta") or {}
        text = fmt_signal_message(
            pair, tf, overall, nwe, conf, reason,
            regime=ind_details.get("regime"),
            trigger=reason if reason.startswith("trend_") else None,
            calibrated_conf=meta.get("calibrated_conf"),
            deriv_note=note)

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
            dump = text + "\n\n" + fmt_brain_dump(decision)
            for attempt in (1, 2):   # one retry (A8): transient TG hiccups
                try:
                    m = await self.bot.send_message(chat_id=self.dev_chat_id, text=dump,
                                                    parse_mode=ParseMode.HTML, disable_web_page_preview=True,
                                                    reply_markup=build_dev_keyboard(session_id))
                    dev_msg_id = m.message_id
                    break
                except Exception as e:
                    logger.error("dev send failed (attempt %d): %s", attempt, e)
            if dev_msg_id is None:
                # No buttons ever reached the dev channel — a button-less
                # ACTIVE session would be unreachable for feedback forever.
                self.store.deactivate_session(session_id)
                logger.error("session %s deactivated (dev send failed twice)", session_id)

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

    if kind == "VERDICT":
        # One tap = ground truth for EVERYONE: all four agents + the brain are
        # graded against it with the active reward map (pending rows), or the
        # policy is netted to this verdict via corrections (auto-graded rows).
        if not sess.get("prediction_id"):
            await q.answer("Prediction still recording — try again in a moment.", show_alert=True)
            return
        verdict = value                                   # buy | sell | skip(=flat)
        # grader.apply_manual_feedback is the SINGLE writer of the session's
        # true_outcome (it sets it from the normalized label on both the
        # pending and correction paths) — the handler must not also write it,
        # or a second tap that returns already_manual would overwrite the
        # stored truth to disagree with what the agents actually trained on.
        result = await asyncio.to_thread(
            grader.apply_manual_feedback, sess["prediction_id"], verdict)
        status = result.get("status")
        if status == "unknown_prediction":                # row vanished — keep session
            await q.answer("Prediction record missing — cannot grade.", show_alert=True)
            return
        store.deactivate_session(session_id)
        await broadcaster.strip_keyboard(sess)
        if status == "already_manual":                    # double tap / second human
            await q.answer("Already graded manually — nothing changed.")
            return
        label = {"buy": "BUY", "sell": "SELL", "skip": "FLAT"}.get(verdict, verdict)
        vals = result.get("rewards") or result.get("corrections") or {}
        parts = " · ".join(f"{k[:4]} {v:+g}" for k, v in vals.items())
        await q.answer(f"Trained vs {label} ({status}) {parts}"[:190])
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
        if not sess.get("prediction_id"):
            # cycle back-fills the link right after record_prediction; a click
            # inside that window (or on an unrepairable legacy session) must
            # NOT burn the session on a no-op grade.
            await q.answer("Prediction still recording — try again in a moment.", show_alert=True)
            return
        news_reward = None if value == "auto" else float(value)
        # to_thread: apply_manual_feedback takes the grader's reward lock and
        # may briefly wait on an in-flight auto grade — never block the loop.
        result = await asyncio.to_thread(
            grader.apply_manual_feedback, sess["prediction_id"], true_outcome,
            news_reward=news_reward)
        store.deactivate_session(session_id)
        await broadcaster.strip_keyboard(sess)
        note = " (news override ignored — agent absent)" if result.get("news_override_ignored") else ""
        await q.answer(f"Feedback applied ({result.get('status')}){note}. Session closed.")


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


async def cmd_derivs(update, context):
    dm = context.application.bot_data["dm"]
    pair = (context.args[0] if context.args else "BTCUSDT").upper()
    out = await asyncio.to_thread(dm.derivatives.decide, pair, "1h")
    if not out.get("available"):
        await _reply(update, f"<b>derivs {pair}</b>\nno USDM future / fetch failed")
        return
    d = out.get("details") or {}
    await _reply(update, f"<b>derivs {pair}</b>\naction={out.get('action')} conf={_r3(out.get('confidence'))}\n"
                         f"funding={d.get('funding_rate')} oiΔ={_r3(d.get('oi_change_pct'))}\n"
                         f"top L/S={_r3(d.get('top_position_ratio'))} acct L/S={_r3(d.get('global_account_ratio'))}")


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
                # timeframes_due speaks UTC (Binance candle boundaries); the
                # IST :30 tick IS the UTC :00 boundary, so only the hour
                # cascade needed the conversion (correctness v3, A1).
                from datetime import timezone as _tz
                tfs = timeframes_due(dt.astimezone(_tz.utc))
                logger.info("cycle at %s for %s", dt.strftime("%Y-%m-%d %H:%M"), tfs)
                # Hourly ingestion BEFORE the cycle so the news agent's RAG
                # grounding includes this hour's headlines.
                try:
                    rag_index = bd.get("rag_index")
                    if rag_index is not None:
                        import time as _time
                        stats = await asyncio.to_thread(
                            ingest_all, rag_index,
                            dedup_window_ts=_time.time() - 7 * 86400)
                        logger.info("news ingest: %s", stats)
                except Exception as e:
                    logger.error("ingest failed: %s", e)
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
    # .env already loaded at module import (see file top) — must precede config
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

    # Membership (Bot D) — fully flag-gated: with MEMBERSHIP_ENABLED=false the
    # package is never imported and the runtime is identical to a build
    # without it.
    import config as _cfg
    membership_app = None
    subs_store = None
    membership_token = os.environ.get("MEMBERSHIP_BOT_TOKEN")
    if _cfg.MEMBERSHIP_ENABLED:
        # The store — and therefore the Pro gate on the control bot — comes up
        # whenever MEMBERSHIP_ENABLED, independent of the storefront token. A
        # missing token must NOT silently leave the six agent commands
        # ungated; it only means the storefront bot can't run.
        from membership.store import SubsStore
        subs_store = SubsStore(_cfg.MEMBERSHIP_DB)
        if not membership_token:
            logger.error("MEMBERSHIP_ENABLED but MEMBERSHIP_BOT_TOKEN is unset — "
                         "Pro gating is ON, but the storefront bot will not start")

    async def post_init(app):
        broadcaster = Broadcaster(app.bot, store, customer, dev)
        app.bot_data.update({"dm": dm, "store": store, "grader": grader, "broadcaster": broadcaster})
        try:
            from rag import RagIndex
            app.bot_data["rag_index"] = RagIndex(store=store)
        except Exception as e:
            logger.error("rag index unavailable: %s", e)
            app.bot_data["rag_index"] = None
        from jobs.nightly import nightly_loop
        app.bot_data["_tasks"] = [asyncio.create_task(scheduler_loop(app)),
                                  asyncio.create_task(grader_loop(app)),
                                  asyncio.create_task(nightly_loop(app))]
        # Each secondary bot starts under its own try/except: one failing to
        # come up must never strand another's poller (an orphaned getUpdates
        # loop 409s on the next restart).
        if control_app is not None:
            try:
                control_app.bot_data["dm"] = dm
                await control_app.initialize()
                await control_app.start()
                await control_app.updater.start_polling(allowed_updates=Update.ALL_TYPES)
                logger.info("control bot started")
            except Exception as e:
                logger.error("control bot failed to start: %s", e)
        if membership_app is not None:
            try:
                await membership_app.initialize()
                await membership_app.start()
                await membership_app.updater.start_polling(allowed_updates=Update.ALL_TYPES)
                me = await membership_app.bot.get_me()
                # publish the storefront username so the Pro-gate refusal on the
                # control bot renders a tappable ?start=pro deep link
                membership_app.bot_data["membership_username"] = me.username
                if control_app is not None:
                    control_app.bot_data["membership_username"] = me.username
                from membership.jobs import membership_loop
                app.bot_data["_tasks"].append(
                    asyncio.create_task(membership_loop(membership_app)))
                logger.info("membership bot started (@%s)", me.username)
            except Exception as e:
                logger.error("membership bot failed to start: %s", e)

    async def post_shutdown(app):
        for t in app.bot_data.get("_tasks", []):
            t.cancel()
        for secondary in (control_app, membership_app):
            if secondary is None:
                continue
            try:
                if secondary.updater and secondary.updater.running:
                    await secondary.updater.stop()
                await secondary.stop()
                await secondary.shutdown()
            except Exception as e:
                logger.error("secondary bot shutdown error: %s", e)

    app = Application.builder().token(token).post_init(post_init).post_shutdown(post_shutdown).build()
    app.add_handler(CallbackQueryHandler(handle_callback))

    if control_token:
        control_app = Application.builder().token(control_token).build()
        _handlers = {"news": cmd_news, "indicator": cmd_indicator,
                     "research": cmd_research, "context": cmd_context,
                     "regime": cmd_regime, "derivs": cmd_derivs}
        if subs_store is not None:
            # Pro gating: subscription + daily fair-use around every command.
            # Flag off -> handlers registered bare, behavior identical to today.
            from membership.gate import requires_pro
            gate = requires_pro(subs_store)
            _handlers = {name: gate(h) for name, h in _handlers.items()}
        for _name, _h in _handlers.items():
            control_app.add_handler(CommandHandler(_name, _h))

    if subs_store is not None and membership_token:
        from membership import bot as membership_bot
        from membership.payments import RazorpayLinks, TronWatcher
        membership_app = Application.builder().token(membership_token).build()
        membership_bot.register(membership_app, subs_store, customer,
                                rzp=RazorpayLinks(), tron=TronWatcher())

    logger.info("starting BitReinforceX runtime")
    app.run_polling(poll_interval=1.0, allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
