#!/usr/bin/env python3
"""
main.py — Orchestrator for BitReinforceX

Aligned to your existing project structure in Projectt/:
- Uses brain/decision_maker.DecisionMaker
- Respects your three child agents and brain policy
- Schedules runs at India candle close (HH:30 IST)
- Parallel analysis for ~50 symbols across timeframes
- Signal filtering rules (NWE vs confidence)
- Telegram broadcasting to Customer & Developer channels
- Developer inline buttons for reinforcement feedback
- Session lifecycle management (auto-close when superseded)

Requirements (add to requirements.txt):
  python-telegram-bot>=21.0

Env vars expected:
  TELEGRAM_BOT_TOKEN="..."
  CUSTOMER_CHAT_ID="-100..."    # channel or group id
  DEV_CHAT_ID="-100..."          # channel or group id

Run:
  python main.py

Notes:
- The brain's interactive feedback (DecisionMaker.feedback) is bypassed; we apply
  learning programmatically when a dev presses inline buttons.
- If OPENAI_API_KEY is missing, NewsAgent will fallback internally; that's fine.
- DataFetcher(prefer_csv=False) will fetch via ccxt when CSVs are unavailable.
"""
from __future__ import annotations
import asyncio
import os
import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple

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
# SYMBOLS = [
#     "AAVE","ADA","ALGO","AR","ARB","ATOM","AVAX","AXS","BCH","BNB",
#     "BTC","CAKE","COMP","CRV","DOGE","DOT","DYDX","ENJ","ETC","ETH",
#     "FET","FIL","FLOW","GALA","GMT","GRT","ICP","IMX","INJ","LINK",
#     "LRC","LUNA","MANA","MKR","NEAR","OP","POL","PYTH","RENDER","SAND",
#     "SHIB","SNX","SOL","STORJ","THETA","UNI","WLD","XRP"
# ]

SYMBOLS = ["SOL"]
SYMBOLS = [s + "USDT" for s in SYMBOLS]

# Timeframes we may schedule
TF_1H = "1h"; TF_4H = "4h"; TF_1D = "1d"; TF_1W = "1w"

# Session expiration (no button pressed):
SESSION_TTL_HOURS = 12

# Concurrency
MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY"))  # tune if you hit rate limits

# ==================
# Global singletons
# ==================
DM = DecisionMaker(prefer_csv=False)

# Active sessions indexed by session_id
SESSIONS: Dict[str, Dict[str, Any]] = {}
# Map (symbol, timeframe) -> latest active session_id to auto-close on supersession
ACTIVE_BY_PAIR_TF: Dict[Tuple[str, str], str] = {}

# ===============
# Utility helpers
# ===============

def now_ist() -> datetime:
    return datetime.now(tz=IST)


def is_candle_close_minute(dt: datetime) -> bool:
    """Candle closes at the 30th minute of every hour in IST per user requirement."""
    return dt.minute == 54


def timeframes_due(dt: datetime) -> List[str]:
    """Return which TFs to run at this close.
    - Every hour @ :30 => run 1h for all symbols
    - Every 4th hour (00,04,08,12,16,20 IST) @ :30 => also run 4h
    - Every day at 00:30 IST => also run 1d
    - Every week on Monday 00:30 IST => also run 1w
    """
    due = [TF_4H]
    # if dt.hour % 4 == 0:
    #     due.append(TF_4H)
    # if dt.hour == 0:  # daily close
    #     due.append(TF_1D)
    #     # Monday daily close (start of week) ⇒ weekly
    #     if dt.weekday() == 0:
    #         due.append(TF_1W)
    return due


def pick_nwe_signal(agent_block: Dict[str, Any]) -> Optional[str]:
    """Extract NWE direct signal from IndicatorAgent output if present."""
    try:
        direct = agent_block["raw"]["details"]["direct_signals"]
        # direct is a list of dicts with keys: name, signal, confidence
        for d in direct:
            if str(d.get("name", "")).lower() == "nwe":
                return str(d.get("signal", "skip")).lower()
    except Exception:
        return None
    return None


def pick_nwe_name_and_conf(agent_block: Dict[str, Any]) -> Tuple[str, Optional[float]]:
    try:
        direct = agent_block["raw"]["details"]["direct_signals"]
        for d in direct:
            if str(d.get("name", "")).lower() == "nwe":
                return str(d.get("signal", "skip")).lower(), float(d.get("confidence", 0.0))
    except Exception:
        pass
    return "skip", None


def should_emit_signal(res: Dict[str, Any]) -> Tuple[bool, str, str, float, str]:
    """Apply user-defined signal rules and return
    (emit, overall_action, nwe_action, confidence, reason)
    reason ∈ {"nwe_direct", "conf_over_80"}
    """
    final_action = str(res["final"]["action"]).lower()
    final_conf = float(res["final"]["confidence"])
    tf = str(res["timeframe"]).lower()

    ind = res["agents"].get("indicator", {})
    nwe_action = pick_nwe_signal(ind) or "skip"

    if tf == TF_1H:
        # 1h: only if direct NWE (non-skip)
        if nwe_action in ("buy", "sell"):
            # If both conditions would be true, 1h still uses NWE
            return True, nwe_action, nwe_action, final_conf, "nwe_direct"
        return False, final_action, nwe_action, final_conf, ""

    # Other TFs: conf>=0.80 OR NWE direct (non-skip)
    conf_hit = final_conf >= 0.80
    nwe_hit = nwe_action in ("buy", "sell")

    if conf_hit and nwe_hit:
        # If conflict, prefer NWE
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
        f"<b>OVERALL TRADE SIGNAL:</b> {overall_action}\n"
        f"<b>NWE SIGNAL:</b> {nwe_action}\n"
        f"💱 <b>PAIR:</b> {pair}\n"
        f"⏰ <b>TIMEFRAME:</b> {tf}\n"
        f"📊 <b>CONFIDENCE:</b> {conf_pc}\n"
        f"🧠 <b>REASON:</b> {reason_text}\n\n"
        "⚠️ <i>Disclaimer: This is NOT financial advice.\n"
        "Trading involves risk — do your own research.\n"
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
    sess = SESSIONS.get(session_id)
    if not sess:
        return
    # Remove keyboards from dev message
    try:
        await app.bot.edit_message_reply_markup(
            chat_id=sess["dev_chat_id"], message_id=sess["dev_msg_id"], reply_markup=None
        )
    except Exception:
        pass
    sess["active"] = False


async def supersede_previous(app: Application, pair: str, tf: str):
    key = (pair, tf)
    prev_id = ACTIVE_BY_PAIR_TF.get(key)
    if prev_id and prev_id in SESSIONS:
        await deactivate_session(app, prev_id)


async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
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
        await q.answer("Closed.")
        await deactivate_session(context.application, session_id)
        return

    if kind == "OUTCOME":
        sess["true_outcome"] = value  # buy/sell/skip
        await q.answer(f"Outcome: {value.upper()}")
        return

    if kind == "REWARD":
        # Reward requires outcome first (unless skip learning)
        true = sess.get("true_outcome", "")
        if not true:
            await q.answer("Select BUY/SELL/SKIP first.", show_alert=True)
            return

        if true == "skip":
            await q.answer("Learning skipped.")
            await deactivate_session(context.application, session_id)
            return

        if value == "auto":
            # +1 if news matched true, else -4
            news_pred = sess["decision"]["agents"].get("news", {}).get("action")
            news_reward = 1.0 if news_pred == true else -4.0
        else:
            try:
                news_reward = float(value)
            except Exception:
                news_reward = -4.0

        # Apply learning to child agents + brain
        await apply_learning(sess, true_outcome=true, news_reward=news_reward)

        await q.answer("Feedback applied. Session closed.")
        await deactivate_session(context.application, session_id)
        return


async def apply_learning(sess: Dict[str, Any], true_outcome: str, news_reward: float):
    """Mirror DecisionMaker.feedback(), but non-interactive."""
    agents = sess["decision"]["agents"]

    # NewsAgent
    try:
        news_pred = agents.get("news", {}).get("action")
        DM.news.learn(action_label=news_pred, reward=news_reward)  # NewsAgent API in your project
    except Exception:
        pass

    # IndicatorAgent
    try:
        ind_pred = agents.get("indicator", {}).get("action")
        DM.indicator.learn(predicted_action=ind_pred, true_outcome=true_outcome)
    except Exception:
        pass

    # ResearchAgent
    try:
        res_pred = agents.get("research", {}).get("action")
        # Some versions accept reward=None
        try:
            DM.research.learn(predicted_action=res_pred, true_outcome=true_outcome, reward=None)
        except Exception:
            DM.research.learn(res_pred, true_outcome)
    except Exception:
        pass

    # Brain policy adjustment
    try:
        # Use DM._apply_feedback_to_brain if accessible; else replicate slowly
        DM._apply_feedback_to_brain(agents, true_outcome, news_reward)  # noqa: SLF001 (intentional)
    except Exception:
        pass


# ==================
# Orchestration Core
# ==================

async def analyse_symbol_tf(symbol: str, tf: str) -> Optional[Dict[str, Any]]:
    try:
        res = DM.decide(symbol, tf)
        if res:
            logger.info(json.dumps({"symbol": symbol, "timeframe": tf, "result": res}))
        return res
    except Exception as e:
        logger.exception(f"Error analysing {symbol} on {tf}: {e}")
        return None


async def run_batch(timeframes: List[str]):
    pairs = SYMBOLS
    sem = asyncio.Semaphore(MAX_CONCURRENCY)

    async def task(sym: str, tf: str) -> Tuple[str, str, Optional[Dict[str, Any]]]:
        async with sem:
            out = await asyncio.to_thread(asyncio.run, asyncio.sleep(0))  # no-op yield
            res = await asyncio.to_thread(DM.decide, sym, tf)
            if res:
                logger.info(json.dumps({"symbol": sym, "timeframe": tf, "result": res}))
            return sym, tf, res

    tasks = [task(sym, tf) for tf in timeframes for sym in pairs]
    for fut in asyncio.as_completed(tasks):
        sym, tf, res = await fut
        if not res:
            continue
        emit, overall, nwe, conf, reason = should_emit_signal(res)
        if not emit:
            continue
        await broadcast_signal(sym, tf, res, overall, nwe, conf, reason)


async def broadcast_signal(pair: str, tf: str, decision: Dict[str, Any],
                           overall: str, nwe: str, conf: float, reason: str):
    app = Application.builder().token(os.environ["TELEGRAM_BOT_TOKEN"]).build()
    # Ensure previous session for (pair, tf) is closed
    await supersede_previous(app, pair, tf)

    text = fmt_signal_message(pair, tf, overall, nwe, conf, reason)

    # Send to customer channel (no buttons)
    cust_chat = int(os.environ["CUSTOMER_CHAT_ID"]) if "CUSTOMER_CHAT_ID" in os.environ else None
    dev_chat = int(os.environ["DEV_CHAT_ID"]) if "DEV_CHAT_ID" in os.environ else None

    cust_msg_id = None
    if cust_chat:
        try:
            m = await app.bot.send_message(chat_id=cust_chat, text=text, parse_mode=ParseMode.HTML, disable_web_page_preview=True)
            cust_msg_id = m.message_id
        except Exception:
            pass

    # Send to developer channel with inline buttons
    dev_msg_id = None
    session_id = str(uuid.uuid4())
    kb = build_dev_keyboard(session_id)
    if dev_chat:
        try:
            m = await app.bot.send_message(chat_id=dev_chat, text=text, parse_mode=ParseMode.HTML,
                                           reply_markup=kb, disable_web_page_preview=True)
            dev_msg_id = m.message_id
        except Exception:
            pass

    # Register session
    sess = {
        "id": session_id,
        "pair": pair,
        "tf": tf,
        "created_at": now_ist().isoformat(),
        "active": True,
        "decision": {
            # Store a compact snapshot for feedback
            "agents": {
                k: {"action": v.get("action"), "confidence": v.get("confidence")}
                for k, v in decision.get("agents", {}).items()
            },
            "final": decision.get("final", {}),
        },
        "true_outcome": "",
        "news_reward": None,
        "cust_chat_id": cust_chat,
        "cust_msg_id": cust_msg_id,
        "dev_chat_id": dev_chat,
        "dev_msg_id": dev_msg_id,
    }
    SESSIONS[session_id] = sess
    ACTIVE_BY_PAIR_TF[(pair, tf)] = session_id

    # Start a lightweight application solely to handle callbacks for this send
    # (We create one per broadcast to keep things simple and stateless here.)
    app.add_handler(CallbackQueryHandler(handle_callback))
    await app.initialize()
    await app.start()
    # Let the app run briefly to accept button presses; in practice, you would
    # have a single long-running bot. Here we sleep a bit then stop; sessions
    # remain in-memory for TTL and supersession handling within this process.
    await asyncio.sleep(2)  # minimal runtime for delivery
    await app.stop()
    await app.shutdown()


async def session_gc():
    """Garbage-collect stale sessions (remove keyboards if still present)."""
    app = Application.builder().token(os.environ["TELEGRAM_BOT_TOKEN"]).build()
    await app.initialize(); await app.start()
    cutoff = now_ist() - timedelta(hours=SESSION_TTL_HOURS)
    for sid, sess in list(SESSIONS.items()):
        if not sess.get("active", False):
            continue
        try:
            created = datetime.fromisoformat(sess["created_at"]).astimezone(IST)
        except Exception:
            created = now_ist() - timedelta(days=1)
        if created < cutoff:
            await deactivate_session(app, sid)
    await app.stop(); await app.shutdown()


# =======================
# Scheduler / Main Runner
# =======================

async def scheduler_loop():
    """Wake every 10s, trigger runs exactly at HH:30 IST."""
    last_run_minute = None
    while True:
        dt = now_ist()
        if is_candle_close_minute(dt) and dt.minute != last_run_minute:
            last_run_minute = dt.minute
            tfs = timeframes_due(dt)
            try:
                await run_batch(tfs)
            except Exception:
                pass
            # clean up sessions occasionally
            try:
                await session_gc()
            except Exception:
                pass
        await asyncio.sleep(180)


def main():
    # Validate Telegram token exists (we still run analysis even if not set, but warn)
    if "TELEGRAM_BOT_TOKEN" not in os.environ:
        print("[WARN] TELEGRAM_BOT_TOKEN not set. Signals will not be sent to Telegram.")
    try:
        asyncio.run(scheduler_loop())
    except KeyboardInterrupt:
        print("Shutting down...")


if __name__ == "__main__":
    main()
