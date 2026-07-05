"""Central configuration for the accuracy upgrade (v2).

Every tunable introduced by the v2 work lives here, read from the environment
with sane defaults, so the Oracle box can flip behaviour via .env without code
changes. Rollout philosophy: each risky change ships behind a flag, defaulted
OFF, and is enabled only after backtest / shadow evidence.

Legacy modules keep their own historical constants (e.g. grader.HORIZON_K);
v2 code imports from here.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, Tuple

# config is the universal chokepoint — EVERY entry point imports it (directly
# or transitively) before reading any flag, and every flag below is snapshotted
# from os.environ at import time. So .env must be loaded HERE, before the first
# os.getenv, or scripts that don't call load_dotenv themselves (run_backtest,
# run_training, refresh_ecosystems, verify_phase1, …) silently run on flag
# DEFAULTS. The tests set BITREINFORCEX_NO_DOTENV=1 (in conftest, before any
# import) to stay hermetic — the .env on a dev/CI box must not leak in.
if not os.getenv("BITREINFORCEX_NO_DOTENV"):
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass


def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def _env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    try:
        return float(v) if v is not None else default
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    try:
        return int(v) if v is not None else default
    except ValueError:
        return default


def _env_json(name: str, default: Any) -> Any:
    v = os.getenv(name)
    if not v:
        return default
    try:
        return json.loads(v)
    except json.JSONDecodeError:
        return default


# --------------------------------------------------------------------------
# Rollout flags (all default OFF — enable on evidence, see plan)
# --------------------------------------------------------------------------
GATE_V2_ENABLED = _env_bool("GATE_V2_ENABLED", False)       # regime-gated signal gate
GATE_1H_TREND = _env_bool("GATE_1H_TREND", False)           # allow trend triggers on 1h
# Backtest-driven amendments (12-pair, 2y, exact-path evidence):
#   trend_reversal graded 0/6 decided; 4h ranging NWE graded 12.5% TB precision.
GATE_TREND_REVERSAL = _env_bool("GATE_TREND_REVERSAL", False)   # supertrend-flip-against-trend emissions
GATE_NWE_HIGHER_TF = _env_bool("GATE_NWE_HIGHER_TF", False)     # NWE emissions on 4h/1d/1w (1h keeps NWE)
TB_GRADING_ENABLED = _env_bool("TB_GRADING_ENABLED", False)  # triple-barrier rewards
DERIVATIVES_ENABLED = _env_bool("DERIVATIVES_ENABLED", False)  # 4th voter
META_SHADOW = _env_bool("META_SHADOW", True)                # stamp meta_p/calibrated_conf
META_GATE_ENABLED = _env_bool("META_GATE_ENABLED", False)   # gate on meta_p

# --------------------------------------------------------------------------
# Correctness v3 (Phase A)
# --------------------------------------------------------------------------
CLOSED_CANDLES_ONLY = _env_bool("CLOSED_CANDLES_ONLY", True)  # drop in-progress candle in live fetches
NWE_EVENT_MODE = _env_bool("NWE_EVENT_MODE", False)           # NWE fires on band CROSSING, not state
NEWS_RAG_ENABLED = _env_bool("NEWS_RAG_ENABLED", True)        # inject stored headlines into news prompts
BRAIN_DEADZONE_V2 = _env_bool("BRAIN_DEADZONE_V2", False)     # directional final needs conf floor
BRAIN_MIN_CONF = _env_float("BRAIN_MIN_CONF", 0.25)

# --------------------------------------------------------------------------
# Indicator agent enhancements (Phase D)
# --------------------------------------------------------------------------
DIVERGENCE_VOTES = _env_bool("DIVERGENCE_VOTES", False)          # RSI/OBV divergence type-2 votes (4h/1d)
EMPIRICAL_DIRECT_CONF = _env_bool("EMPIRICAL_DIRECT_CONF", False)  # EB win-rate confidences for direct signals
INDICATOR_CONF_PATH = os.getenv("INDICATOR_CONF_PATH", "logs/indicator_conf.json")

# v3.4 extra type-2 confluence votes — csv of keys, each backtest-gated before
# default promotion (see docs). Unknown keys are ignored.
_T2_VOTE_KEYS = frozenset({"rsi30", "mfi", "cci", "vwap", "fib", "ichimoku"})
T2_EXTRA_VOTES = frozenset(
    s.strip().lower() for s in os.getenv("T2_EXTRA_VOTES", "").split(",") if s.strip()
) & _T2_VOTE_KEYS
T2_RULE_LEARNING = _env_bool("T2_RULE_LEARNING", False)   # per-rule learned type-2 weights

# --------------------------------------------------------------------------
# News agent enhancements (Phase C)
# --------------------------------------------------------------------------
NEWS_EVENTS_ENABLED = _env_bool("NEWS_EVENTS_ENABLED", False)   # typed event extraction

# --------------------------------------------------------------------------
# Research agent enhancements (Phase B)
# --------------------------------------------------------------------------
MACRO_PRICES_ENABLED = _env_bool("MACRO_PRICES_ENABLED", False)  # real SPX/DXY trends into logics 2/5
MONEY_FLOW_V2 = _env_bool("MONEY_FLOW_V2", False)                # quantitative 4-phase money-flow
ECOSYSTEMS_AUTO = _env_bool("ECOSYSTEMS_AUTO", False)            # CoinGecko-refreshed ecosystem lists
ECOSYSTEMS_CACHE_PATH = os.getenv("ECOSYSTEMS_CACHE_PATH", "logs/ecosystems_cache.json")

# --------------------------------------------------------------------------
# Signal gate
# --------------------------------------------------------------------------
CONFIDENCE_GATE = _env_float("CONFIDENCE_GATE", 0.80)
VOLUME_SMA_LEN = _env_int("VOLUME_SMA_LEN", 20)             # vol_ok = vol > SMA20(vol)

# --------------------------------------------------------------------------
# Regime classifier (ADX / Choppiness / realized-vol percentile, hysteresis)
# --------------------------------------------------------------------------
REGIME_ADX_LEN = _env_int("REGIME_ADX_LEN", 14)
REGIME_CHOP_LEN = _env_int("REGIME_CHOP_LEN", 14)
REGIME_VOL_LOOKBACK = _env_int("REGIME_VOL_LOOKBACK", 100)   # percentile window
REGIME_ADX_ENTER = _env_float("REGIME_ADX_ENTER", 25.0)      # enter trending
REGIME_ADX_EXIT = _env_float("REGIME_ADX_EXIT", 20.0)        # exit trending (AND chop)
REGIME_CHOP_ENTER = _env_float("REGIME_CHOP_ENTER", 38.2)    # chop <= enter -> trending
REGIME_CHOP_EXIT = _env_float("REGIME_CHOP_EXIT", 45.0)      # chop >= exit (AND adx) -> leave
REGIME_MIN_DWELL = _env_int("REGIME_MIN_DWELL", 2)           # bars before a flip allows another
REGIME_WALK_BARS = _env_int("REGIME_WALK_BARS", 50)          # state-machine walk length

# --------------------------------------------------------------------------
# Triple-barrier grading
# --------------------------------------------------------------------------
ATR_LEN = _env_int("ATR_LEN", 14)
# per-TF (tp_mult, sl_mult) in ATR units
_BARRIER_DEFAULTS: Dict[str, Tuple[float, float]] = {
    "1h": (1.5, 1.0), "4h": (1.5, 1.0), "1d": (1.5, 1.0), "1w": (2.0, 1.25),
}


def _parse_barrier_mults(raw: dict) -> Dict[str, Tuple[float, float]]:
    """Validate BARRIER_MULTS entries: each must be a 2-sequence of numbers.
    A JSON string like "1.5,1.0" would otherwise become a tuple of CHARACTERS
    and blow up as a TypeError deep inside barrier_prices — fall back to the
    per-TF default with a loud warning instead."""
    out: Dict[str, Tuple[float, float]] = {}
    for tf in set(raw) | set(_BARRIER_DEFAULTS):
        v = raw.get(tf, _BARRIER_DEFAULTS.get(tf))
        try:
            if isinstance(v, (str, bytes)) or len(v) != 2:
                raise ValueError("need a [tp_mult, sl_mult] pair")
            out[tf] = (float(v[0]), float(v[1]))
        except Exception:
            default = _BARRIER_DEFAULTS.get(tf)
            print(f"[config] BARRIER_MULTS[{tf!r}]={v!r} invalid; "
                  f"using default {default}", file=sys.stderr)
            if default is not None:
                out[tf] = default
    return out


BARRIER_MULTS: Dict[str, Tuple[float, float]] = _parse_barrier_mults(
    _env_json("BARRIER_MULTS", _BARRIER_DEFAULTS))
# Reward map v2 (used when TB_GRADING_ENABLED)
REWARD_CORRECT = _env_float("REWARD_CORRECT", 1.0)
REWARD_WRONG = _env_float("REWARD_WRONG", -4.0)
REWARD_TIMEOUT_FLAT = _env_float("REWARD_TIMEOUT_FLAT", -1.5)   # directional call, market went nowhere
REWARD_MISSED_MOVE = _env_float("REWARD_MISSED_MOVE", -1.0)     # predicted skip, market moved

# --------------------------------------------------------------------------
# Derivatives agent (Binance USDM public data)
# --------------------------------------------------------------------------
DERIV_TTL_SECONDS = _env_int("DERIV_TTL_SECONDS", 55 * 60)
DERIV_FUNDING_EXTREME = _env_float("DERIV_FUNDING_EXTREME", 0.0005)  # 0.05% per 8h
DERIV_OI_WINDOW_H = _env_int("DERIV_OI_WINDOW_H", 6)
DERIV_BRAIN_SCORE = _env_float("DERIV_BRAIN_SCORE", 1.5)

# --------------------------------------------------------------------------
# Sentiment agent (v3.5): Fear&Greed + free BTC on-chain + taker order-flow
# --------------------------------------------------------------------------
SENTIMENT_ENABLED = _env_bool("SENTIMENT_ENABLED", False)     # 5th voter (shadow at go-live)
SENTIMENT_TTL_SECONDS = _env_int("SENTIMENT_TTL_SECONDS", 55 * 60)  # market-wide bundle cache
SENTIMENT_BRAIN_SCORE = _env_float("SENTIMENT_BRAIN_SCORE", 1.5)

# --------------------------------------------------------------------------
# Meta-labeling / calibration (nightly)
# --------------------------------------------------------------------------
META_MIN_ROWS = _env_int("META_MIN_ROWS", 500)
META_MODEL = os.getenv("META_MODEL", "logreg")               # logreg | lgbm
META_GATE_THRESHOLD = _env_float("META_GATE_THRESHOLD", 0.55)
CALIBRATION_MIN_ROWS_PER_TF = _env_int("CALIBRATION_MIN_ROWS_PER_TF", 150)
NIGHTLY_HOUR_IST = _env_int("NIGHTLY_HOUR_IST", 2)
META_MODEL_PATH = os.getenv("META_MODEL_PATH", "logs/meta_model.pkl")
META_METRICS_PATH = os.getenv("META_METRICS_PATH", "logs/meta_metrics.json")
CALIBRATION_PATH = os.getenv("CALIBRATION_PATH", "logs/calibration.json")

# --------------------------------------------------------------------------
# Membership / subscriptions (Bot D). Default OFF: runtime identical to a
# build without the membership package. Payment credentials are read from the
# environment AT CALL TIME (never cached here) — they are secrets.
# --------------------------------------------------------------------------
MEMBERSHIP_ENABLED = _env_bool("MEMBERSHIP_ENABLED", False)
MEMBERSHIP_DB = os.getenv("MEMBERSHIP_DB", "logs/subscriptions.db")
def _parse_admin_ids(raw: str) -> frozenset:
    """Crash-proof: one malformed id must not kill the whole app at import."""
    out = set()
    for x in raw.replace(" ", "").split(","):
        if not x:
            continue
        try:
            out.add(int(x))
        except ValueError:
            print(f"[config] ADMIN_USER_IDS entry {x!r} is not a numeric id — ignored",
                  file=sys.stderr)
    return frozenset(out)


ADMIN_USER_IDS = _parse_admin_ids(os.getenv("ADMIN_USER_IDS", ""))
PRO_DAILY_QUERY_CAP = _env_int("PRO_DAILY_QUERY_CAP", 30)
MEMBERSHIP_GRACE_HOURS = _env_float("MEMBERSHIP_GRACE_HOURS", 24.0)
REFERRAL_BONUS_DAYS = _env_int("REFERRAL_BONUS_DAYS", 7)
