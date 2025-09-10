#!/usr/bin/env python3
"""
main.py – Optimized Orchestrator for BitReinforceX

Key improvements:
- Fixed the PTBUserWarning by properly using post_init
- Reduced sleep time from 1000s to 30s for responsive scheduling
- Added proper task cancellation and cleanup
- Improved concurrency with better error handling
- Added caching to prevent redundant API calls
- Better session management and cleanup
"""
from __future__ import annotations
import asyncio
import os
import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from functools import lru_cache
import time

# Local timezone handling (IST)
from zoneinfo import ZoneInfo
IST = ZoneInfo("Asia/Kolkata")

# Project imports (relative to this main.py)
from brain.decision_maker import DecisionMaker

# Telegram (async)
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ParseMode
from telegram.ext import Application, CallbackQueryHandler, ContextTypes

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("main")

# File handler for results log
results_handler = logging.FileHandler("results.log")
results_handler.setLevel(logging.INFO)
results_formatter = logging.Formatter("%(asctime)s %(message)s")
results_handler.setFormatter(results_formatter)
logger.addHandler(results_handler)

# =====================
# Configurable settings
# =====================
# Symbols to analyse (USDT-margined on Binance)
SYMBOLS = [
    "AAVE","ADA","ALGO","AR","ARB","ATOM","AVAX","AXS","BCH","BNB",
    "BTC","CAKE","COMP","CRV","DOGE","DOT","DYDX","ENJ","ETC","ETH",
    "FET","FIL","FLOW","GALA","GMT","GRT","ICP","IMX","INJ","LINK",
    "LRC","LUNA","MANA","MKR","NEAR","OP","POL","PYTH","RENDER","SAND",
    "SHIB","SNX","SOL","STORJ","THETA","UNI","WLD","XRP"
]
# SYMBOLS = ["TAO", "BTC"]
SYMBOLS = [s + "USDT" for s in SYMBOLS]

# Timeframes
TF_1H = "1h"; TF_4H = "4h"; TF_1D = "1d"; TF_1W = "1w"

# Session expiration (no button pressed):
SESSION_TTL_HOURS = 12

# Concurrency - set a reasonable default if not in env
MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY", "5"))

# Cache TTL for decision results (prevent redundant API calls)
DECISION_CACHE_TTL = 300  # 5 minutes

# ==================
# Global singletons
# ==================
DM = DecisionMaker(prefer_csv=False)

# Active sessions indexed by session_id
SESSIONS: Dict[str, Dict[str, Any]] = {}
# Map (symbol, timeframe) -> latest active session_id to auto-close on supersession
ACTIVE_BY_PAIR_TF: Dict[Tuple[str, str], str] = {}
# Decision cache to prevent redundant API calls
DECISION_CACHE: Dict[Tuple[str, str], Tuple[float, Dict]] = {}
# Track scheduler task for proper cleanup
SCHEDULER_TASK: Optional[asyncio.Task] = None

# ===============
# Utility helpers
# ===============

def now_ist() -> datetime:
    return datetime.now(tz=IST)


def is_candle_close_minute(dt: datetime) -> bool:
    """For testing, returns True. In production, use dt.minute == 30"""
    # Uncomment for production:
    return dt.minute == 30
    # return True  # For testing


def timeframes_due(dt: datetime) -> List[str]:
    """Return which TFs to run at this close."""
    # For testing, just return 4h
    # due = [TF_4H]
    
    # Uncomment for production:
    due = []
    if dt.hour % 4 == 0:
        due.append(TF_4H)
    if dt.hour == 0:  # daily close
        due.append(TF_1D)
        if dt.weekday() == 0:  # Monday
            due.append(TF_1W)
    
    return due


def pick_nwe_signal(agent_block: Dict[str, Any]) -> Optional[str]:
    """Extract NWE direct signal from IndicatorAgent output if present."""
    try:
        direct = agent_block["raw"]["details"]["direct_signals"]
        for d in direct:
            if str(d.get("name", "")).lower() == "nwe":
                return str(d.get("signal", "skip")).lower()
    except Exception:
        return None
    return None


def should_emit_signal(res: Dict[str, Any]) -> Tuple[bool, str, str, float, str]:
    """Apply user-defined signal rules."""
    if not res:
        return False, "skip", "skip", 0.0, ""
        
    final_action = str(res.get("final", {}).get("action", "skip")).lower()
    final_conf = float(res.get("final", {}).get("confidence", 0.0))
    tf = str(res.get("timeframe", "")).lower()

    ind = res.get("agents", {}).get("indicator", {})
    nwe_action = pick_nwe_signal(ind) or "skip"

    if tf == TF_1H:
        # 1h: only if direct NWE (non-skip)
        if nwe_action in ("buy", "sell"):
            return True, nwe_action, nwe_action, final_conf, "nwe_direct"
        return False, final_action, nwe_action, final_conf, ""

    # Other TFs: conf>=0.10 OR NWE direct (non-skip)
    conf_hit = final_conf >= 0.80
    nwe_hit = nwe_action in ("buy", "sell")

    if conf_hit and nwe_hit:
        overall = nwe_action
        reason = "nwe_direct"
        return True, overall, nwe_action, final_conf, reason
    if nwe_hit:
        return True, nwe_action, nwe_action, final_conf, "nwe_direct"
    if conf_hit:
        return True, final_action, nwe_action, final_conf, "conf_over_80"

    return False, final_action, nwe_action, final_conf, ""


def fmt_signal_message(pair: str, tf: str, overall_action: str, nwe_action: str,
                       conf: float, reason: str) -> str:
    reason_text = "NWE" if reason == "nwe_direct" else "Confidence > 80%"
    conf_pc = f"{conf*100:.2f}%"
    msg = (
        "<b>🚨 SIGNAL ALERT 🚨</b>\n\n"
        f"<b>OVERALL TRADE SIGNAL:</b> {overall_action.upper()}\n"
        f"<b>NWE SIGNAL:</b> {nwe_action.upper()}\n"
        f"💱 <b>PAIR:</b> {pair}\n"
        f"⏰ <b>TIMEFRAME:</b> {tf}\n"
        f"📊 <b>CONFIDENCE:</b> {conf_pc}\n"
        f"🧠 <b>REASON:</b> {reason_text}\n\n"
        "⚠️ <i>Disclaimer: This is NOT financial advice.\n"
        "Trading involves risk – do your own research.\n"
        "Sharing or reselling these signals is illegal.</i>\n\n"
        "~ <b>BitReinforceX</b>\n  \"Reinforcing your trades with AI power\""
    )
    return msg


# =======================
# Telegram: Inline Buttons
# =======================

def build_dev_keyboard(session_id: str) -> InlineKeyboardMarkup:
    row1 = [
        InlineKeyboardButton(text="BUY", callback_data=f"{session_id}|OUTCOME|buy"),
        InlineKeyboardButton(text="SELL", callback_data=f"{session_id}|OUTCOME|sell"),
        InlineKeyboardButton(text="SKIP LEARNING", callback_data=f"{session_id}|OUTCOME|skip"),
    ]
    row2 = [
        InlineKeyboardButton(text="1.0", callback_data=f"{session_id}|REWARD|1.0"),
        InlineKeyboardButton(text="-4.0", callback_data=f"{session_id}|REWARD|-4.0"),
        InlineKeyboardButton(text="Auto-Assign", callback_data=f"{session_id}|REWARD|auto"),
    ]
    row3 = [InlineKeyboardButton(text="CLOSE SESSION", callback_data=f"{session_id}|CLOSE|x")]
    return InlineKeyboardMarkup([row1, row2, row3])


async def deactivate_session(app: Application, session_id: str):
    """Deactivate a session and remove its inline keyboard."""
    sess = SESSIONS.get(session_id)
    if not sess:
        return
    
    # Remove keyboards from dev message
    try:
        if sess.get("dev_chat_id") and sess.get("dev_msg_id"):
            await app.bot.edit_message_reply_markup(
                chat_id=sess["dev_chat_id"], 
                message_id=sess["dev_msg_id"], 
                reply_markup=None
            )
    except Exception as e:
        logger.debug(f"Could not remove keyboard for session {session_id}: {e}")
    
    sess["active"] = False


async def supersede_previous(app: Application, pair: str, tf: str):
    """Close previous session for the same pair/timeframe."""
    key = (pair, tf)
    prev_id = ACTIVE_BY_PAIR_TF.get(key)
    if prev_id and prev_id in SESSIONS:
        await deactivate_session(app, prev_id)


async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle inline button callbacks."""
    if not update.callback_query:
        return
    
    q = update.callback_query
    try:
        session_id, kind, value = q.data.split("|", 2)
    except Exception:
        await q.answer("Malformed callback data.", show_alert=True)
        return

    sess = SESSIONS.get(session_id)
    if not sess or not sess.get("active", False):
        await q.answer("Session is not active.")
        return

    if kind == "CLOSE":
        await q.answer("Session closed.")
        await deactivate_session(context.application, session_id)
        return

    if kind == "OUTCOME":
        sess["true_outcome"] = value
        await q.answer(f"Outcome set: {value.upper()}")
        return

    if kind == "REWARD":
        true = sess.get("true_outcome", "")
        if not true:
            await q.answer("Please select BUY/SELL/SKIP first.", show_alert=True)
            return

        if true == "skip":
            await q.answer("Learning skipped. Session closed.")
            await deactivate_session(context.application, session_id)
            return

        if value == "auto":
            news_pred = sess["decision"]["agents"].get("news", {}).get("action")
            news_reward = 1.0 if news_pred == true else -4.0
        else:
            try:
                news_reward = float(value)
            except Exception:
                news_reward = -4.0

        await apply_learning(sess, true_outcome=true, news_reward=news_reward)
        await q.answer("Feedback applied. Session closed.")
        await deactivate_session(context.application, session_id)


async def apply_learning(sess: Dict[str, Any], true_outcome: str, news_reward: float):
    """Apply reinforcement learning feedback."""
    agents = sess["decision"]["agents"]

    # NewsAgent
    try:
        news_pred = agents.get("news", {}).get("action")
        if news_pred:
            DM.news.learn(action_label=news_pred, reward=news_reward)
    except Exception as e:
        logger.debug(f"NewsAgent learning failed: {e}")

    # IndicatorAgent
    try:
        ind_pred = agents.get("indicator", {}).get("action")
        if ind_pred:
            DM.indicator.learn(predicted_action=ind_pred, true_outcome=true_outcome)
    except Exception as e:
        logger.debug(f"IndicatorAgent learning failed: {e}")

    # ResearchAgent
    try:
        res_pred = agents.get("research", {}).get("action")
        if res_pred:
            try:
                DM.research.learn(predicted_action=res_pred, true_outcome=true_outcome, reward=None)
            except Exception:
                DM.research.learn(res_pred, true_outcome)
    except Exception as e:
        logger.debug(f"ResearchAgent learning failed: {e}")

    # Brain policy adjustment
    try:
        DM._apply_feedback_to_brain(agents, true_outcome, news_reward)
    except Exception as e:
        logger.debug(f"Brain feedback failed: {e}")


# ==================
# Orchestration Core
# ==================

async def get_decision_cached(symbol: str, tf: str) -> Optional[Dict[str, Any]]:
    """Get decision with caching to prevent redundant API calls."""
    cache_key = (symbol, tf)
    now = time.time()
    
    # Check cache
    if cache_key in DECISION_CACHE:
        cached_time, cached_result = DECISION_CACHE[cache_key]
        if now - cached_time < DECISION_CACHE_TTL:
            logger.debug(f"Using cached result for {symbol} {tf}")
            return cached_result
    
    # Fetch new result
    try:
        result = await asyncio.to_thread(DM.decide, symbol, tf)
        DECISION_CACHE[cache_key] = (now, result)
        return result
    except Exception as e:
        logger.error(f"Error getting decision for {symbol} {tf}: {e}")
        return None


async def run_batch(app: Optional[Application], timeframes: List[str]):
    """Run analysis batch with improved concurrency."""
    pairs = SYMBOLS
    sem = asyncio.Semaphore(MAX_CONCURRENCY)
    
    async def analyze_pair(sym: str, tf: str):
        async with sem:
            try:
                res = await get_decision_cached(sym, tf)
                if res:
                    logger.info(f"Analyzed {sym} {tf}: {res.get('final', {}).get('action', 'skip')}")
                    emit, overall, nwe, conf, reason = should_emit_signal(res)
                    if emit:
                        await broadcast_signal(app, sym, tf, overall, nwe, conf, reason)
                return sym, tf, res
            except Exception as e:
                logger.error(f"Failed to analyze {sym} {tf}: {e}")
                return sym, tf, None
    
    # Create all tasks
    tasks = [analyze_pair(sym, tf) for tf in timeframes for sym in pairs]
    
    # Execute with timeout
    try:
        results = await asyncio.gather(*tasks, return_exceptions=True)
        successful = sum(1 for _, _, res in results if res and not isinstance(res, Exception))
        logger.info(f"Batch complete: {successful}/{len(tasks)} successful")
    except Exception as e:
        logger.error(f"Batch execution failed: {e}")


async def broadcast_signal(app: Optional[Application], pair: str, tf: str, 
                          overall: str, nwe: str, conf: float, reason: str):
    """Broadcast signal to Telegram channels."""
    cust_chat = os.environ.get("CUSTOMER_CHAT_ID")
    dev_chat = os.environ.get("DEV_CHAT_ID")
    
    if not (cust_chat or dev_chat):
        logger.warning("No chat IDs configured for broadcasting")
        return
    
    text = fmt_signal_message(pair, tf, overall, nwe, conf, reason)
    session_id = str(uuid.uuid4())
    
    # Get fresh decision for session (use cache if available)
    snapshot = await get_decision_cached(pair, tf) or {"agents": {}, "final": {}}
    
    sess = {
        "id": session_id,
        "pair": pair,
        "tf": tf,
        "created_at": now_ist().isoformat(),
        "active": True,
        "decision": {
            "agents": {
                k: {"action": v.get("action"), "confidence": v.get("confidence")} 
                for k, v in snapshot.get("agents", {}).items()
            },
            "final": snapshot.get("final", {}),
        },
        "true_outcome": "",
        "news_reward": None,
        "cust_chat_id": int(cust_chat) if cust_chat else None,
        "cust_msg_id": None,
        "dev_chat_id": int(dev_chat) if dev_chat else None,
        "dev_msg_id": None,
    }
    
    if app:
        # Supersede previous session for this pair/tf
        await supersede_previous(app, pair, tf)
        
        # Send to customer channel
        if cust_chat:
            try:
                m = await app.bot.send_message(
                    chat_id=int(cust_chat), 
                    text=text, 
                    parse_mode=ParseMode.HTML, 
                    disable_web_page_preview=True
                )
                sess["cust_msg_id"] = m.message_id
            except Exception as e:
                logger.error(f"Failed to send customer message: {e}")
        
        # Send to dev channel with buttons
        if dev_chat:
            try:
                kb = build_dev_keyboard(session_id)
                m = await app.bot.send_message(
                    chat_id=int(dev_chat), 
                    text=text, 
                    parse_mode=ParseMode.HTML, 
                    reply_markup=kb, 
                    disable_web_page_preview=True
                )
                sess["dev_msg_id"] = m.message_id
            except Exception as e:
                logger.error(f"Failed to send dev message: {e}")
    
    SESSIONS[session_id] = sess
    ACTIVE_BY_PAIR_TF[(pair, tf)] = session_id
    logger.info(f"Signal broadcast for {pair} {tf}: {overall.upper()}")


async def session_gc(app: Optional[Application]):
    """Garbage collect expired sessions."""
    cutoff = now_ist() - timedelta(hours=SESSION_TTL_HOURS)
    expired_count = 0
    
    for sid, sess in list(SESSIONS.items()):
        if not sess.get("active", False):
            continue
            
        try:
            created = datetime.fromisoformat(sess["created_at"]).astimezone(IST)
        except Exception:
            created = now_ist() - timedelta(days=1)
            
        if created < cutoff:
            if app:
                await deactivate_session(app, sid)
            else:
                sess["active"] = False
            expired_count += 1
    
    if expired_count > 0:
        logger.info(f"Cleaned up {expired_count} expired sessions")


async def clear_cache():
    """Clear decision cache periodically."""
    global DECISION_CACHE
    now = time.time()
    expired = [k for k, (t, _) in DECISION_CACHE.items() if now - t > DECISION_CACHE_TTL]
    for k in expired:
        del DECISION_CACHE[k]
    if expired:
        logger.debug(f"Cleared {len(expired)} cached decisions")


# =======================
# Scheduler / Main Runner
# =======================

async def scheduler_loop(app: Optional[Application]):
    """Main scheduler loop with improved efficiency."""
    last_run_ts = None
    logger.info("Scheduler started")
    
    while True:
        try:
            dt = now_ist().replace(second=0, microsecond=0) 
            
            # Check if it's time to run
            if is_candle_close_minute(dt) and dt != last_run_ts:
                last_run_ts = dt
                tfs = timeframes_due(dt)
                print(dt)
                logger.info(f"Running batch at {dt.strftime('%Y-%m-%d %H:%M:%S')} for timeframes: {tfs}")
                
                # Run analysis batch
                await run_batch(app, tfs)
                
                # Clean up sessions and cache
                await session_gc(app)
                await clear_cache()
            
            # Sleep for 30 seconds (more responsive than 1000s)
            await asyncio.sleep(30)
            
        except asyncio.CancelledError:
            logger.info("Scheduler loop cancelled")
            break
        except Exception as e:
            logger.error(f"Scheduler loop error: {e}")
            await asyncio.sleep(60)  # Wait a bit longer on error


async def post_init(application: Application) -> None:
    """Post-initialization callback for the Telegram application."""
    global SCHEDULER_TASK
    # Create the scheduler task properly after application is initialized
    SCHEDULER_TASK = asyncio.create_task(scheduler_loop(application))
    logger.info("Scheduler task created in post_init")


async def post_shutdown(application: Application) -> None:
    """Cleanup callback for graceful shutdown."""
    global SCHEDULER_TASK
    if SCHEDULER_TASK and not SCHEDULER_TASK.done():
        SCHEDULER_TASK.cancel()
        try:
            await SCHEDULER_TASK
        except asyncio.CancelledError:
            pass
    logger.info("Scheduler task cancelled in post_shutdown")


def main():
    """Main entry point."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    
    if token:
        # Build application with proper initialization
        app = (
            Application.builder()
            .token(token)
            .post_init(post_init)
            .post_shutdown(post_shutdown)
            .build()
        )
        
        # Add callback handler
        app.add_handler(CallbackQueryHandler(handle_callback))
        
        logger.info("Starting Telegram bot with integrated scheduler")
        
        # Run polling - this blocks until shutdown
        app.run_polling(poll_interval=1.0, allowed_updates=Update.ALL_TYPES)
        
    else:
        # No Telegram token - run scheduler standalone
        logger.warning("TELEGRAM_BOT_TOKEN not set. Running scheduler without Telegram.")
        try:
            asyncio.run(scheduler_loop(None))
        except KeyboardInterrupt:
            logger.info("Scheduler stopped by user")


if __name__ == "__main__":
    main()