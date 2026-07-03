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
from typing import Any, Dict, Tuple


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
BARRIER_MULTS: Dict[str, Tuple[float, float]] = {
    tf: tuple(v) for tf, v in _env_json("BARRIER_MULTS", {
        "1h": (1.5, 1.0), "4h": (1.5, 1.0), "1d": (1.5, 1.0), "1w": (2.0, 1.25),
    }).items()
}
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
