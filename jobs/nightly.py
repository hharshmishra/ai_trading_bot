"""Nightly in-process training (accuracy upgrade Phase 5).

Third background asyncio task beside the scheduler and grader. At
NIGHTLY_HOUR_IST it trains, from the system's OWN graded predictions:

1. **Meta-label model** — p(final call correct | prediction-time features).
   LogisticRegression (balanced classes) behind a scaler; LightGBM only if
   installed and META_MODEL=lgbm. Needs >= META_MIN_ROWS graded directional
   rows; the last 20% (time-ordered, never shuffled) is the holdout. Artifact:
   logs/meta_model.pkl + logs/meta_metrics.json.

2. **Confidence calibration** — per-TF isotonic regression mapping the brain's
   raw final_confidence to observed hit rate, exported as JSON knots applied
   at runtime with np.interp — the inference path needs ZERO sklearn.

Shadow-first: cycle stamps meta_p / calibrated_conf on every prediction; the
meta gate only acts when META_GATE_ENABLED (criteria: holdout AUC >= 0.60 and
precision lift >= +5pts over >= 4 weeks of shadow rows).
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

import numpy as np

import config
from jobs.features import FEATURE_NAMES, meta_features_from_prediction_row

logger = logging.getLogger("bitreinforcex.nightly")
IST = ZoneInfo("Asia/Kolkata")

_MODEL_CACHE: Dict[str, Any] = {"path_mtime": None, "model": None}
_CALIB_CACHE: Dict[str, Any] = {"path_mtime": None, "knots": None}


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def _directional(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [r for r in rows if (r.get("final_action") or "").lower() in ("buy", "sell")]


def _label_correct(r: Dict[str, Any]) -> int:
    return int((r.get("final_action") or "").lower() == (r.get("realized_label") or "").lower())


def train_meta_model(rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Fit the meta-label model; returns holdout metrics or None if not enough
    data. Time-ordered split — random shuffles leak future regimes into train."""
    rows = _directional(rows)
    if len(rows) < config.META_MIN_ROWS:
        logger.info("meta: %d directional graded rows < %d — skipping",
                    len(rows), config.META_MIN_ROWS)
        return None

    X = np.array([meta_features_from_prediction_row(r) for r in rows], dtype=float)
    y = np.array([_label_correct(r) for r in rows], dtype=int)
    cut = int(len(rows) * 0.8)
    if y[:cut].sum() in (0, cut):   # degenerate train labels
        logger.info("meta: degenerate labels — skipping")
        return None

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    model = None
    if config.META_MODEL == "lgbm":
        try:
            from lightgbm import LGBMClassifier
            model = LGBMClassifier(n_estimators=200, class_weight="balanced")
        except ImportError:
            logger.info("meta: lightgbm not installed — falling back to logreg")
    if model is None:
        model = make_pipeline(StandardScaler(),
                              LogisticRegression(class_weight="balanced", max_iter=1000))

    model.fit(X[:cut], y[:cut])
    p_hold = model.predict_proba(X[cut:])[:, 1]
    y_hold = y[cut:]

    auc = None
    if 0 < y_hold.sum() < len(y_hold):
        auc = float(roc_auc_score(y_hold, p_hold))
    base_rate = float(y_hold.mean()) if len(y_hold) else None
    gated = y_hold[p_hold >= config.META_GATE_THRESHOLD]
    gated_precision = float(gated.mean()) if len(gated) else None
    lift = (gated_precision - base_rate) if (gated_precision is not None
                                             and base_rate is not None) else None

    import joblib
    os.makedirs(os.path.dirname(config.META_MODEL_PATH) or ".", exist_ok=True)
    joblib.dump({"model": model, "feature_names": FEATURE_NAMES}, config.META_MODEL_PATH)

    metrics = {"trained_ts": time.time(), "n_train": cut, "n_holdout": len(rows) - cut,
               "holdout_auc": auc, "holdout_base_rate": base_rate,
               "gate_threshold": config.META_GATE_THRESHOLD,
               "gated_precision": gated_precision, "precision_lift": lift,
               "model": type(model).__name__}
    with open(config.META_METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    _MODEL_CACHE["path_mtime"] = None   # force reload on next inference
    logger.info("meta: trained on %d, holdout AUC=%s lift=%s", cut, auc, lift)
    return metrics


def fit_calibration(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Per-TF isotonic knots {tf: {"x": [...], "y": [...]}} written to
    CALIBRATION_PATH. TFs with < CALIBRATION_MIN_ROWS_PER_TF rows are omitted
    (identity mapping at runtime)."""
    rows = _directional(rows)
    knots: Dict[str, Any] = {}
    from sklearn.isotonic import IsotonicRegression
    for tf in ("1h", "4h", "1d", "1w"):
        sub = [r for r in rows if (r.get("tf") or "").lower() == tf]
        if len(sub) < config.CALIBRATION_MIN_ROWS_PER_TF:
            continue
        x = np.array([float(r.get("final_confidence") or 0.0) for r in sub])
        y = np.array([_label_correct(r) for r in sub], dtype=float)
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        iso.fit(x, y)
        knots[tf] = {"x": [float(v) for v in iso.X_thresholds_],
                     "y": [float(v) for v in iso.y_thresholds_]}
    payload = {"fitted_ts": time.time(), "knots": knots}
    with open(config.CALIBRATION_PATH, "w") as f:
        json.dump(payload, f, indent=2)
    _CALIB_CACHE["path_mtime"] = None
    logger.info("calibration: fitted for %s", sorted(knots))
    return payload


def fit_direct_conf(rows: List[Dict[str, Any]], m: int = 20,
                    min_n: int = 30) -> Dict[str, Any]:
    """Empirical-Bayes per-direct-indicator confidence (enhancement D4).

    Groups graded rows by which direct signal fired (blend.fired_direct), wins
    = indicator_action matched the realized label, shrunk toward the global
    directional win rate: p̂ = (wins + m·p̄) / (n + m). Confidence mapping
    conf = 0.5 + 0.45·p̂ keeps the [0.5, 0.95] range the merge code expects.
    Indicators with n < min_n get no entry (callers keep their default).
    Writes config.INDICATOR_CONF_PATH.
    """
    directional = [r for r in rows
                   if (r.get("indicator_action") or "").lower() in ("buy", "sell")
                   and isinstance(r.get("indicator_blend"), dict)
                   and r["indicator_blend"].get("fired_direct")]
    total = len(directional)
    if total:
        global_wins = sum(1 for r in directional
                          if (r["indicator_action"] or "").lower() == (r.get("realized_label") or "").lower())
        p_bar = global_wins / total
    else:
        p_bar = 0.5

    by_name: Dict[str, List[int]] = {}
    for r in directional:
        name = r["indicator_blend"]["fired_direct"]
        win = int((r["indicator_action"] or "").lower() == (r.get("realized_label") or "").lower())
        by_name.setdefault(name, []).append(win)

    conf: Dict[str, Any] = {}
    for name, wins in by_name.items():
        n = len(wins)
        if n < min_n:
            continue
        shrunk = (sum(wins) + m * p_bar) / (n + m)
        conf[name] = {"n": n, "win_rate": sum(wins) / n,
                      "shrunk": shrunk, "conf": round(0.5 + 0.45 * shrunk, 4)}

    payload = {"fitted_ts": time.time(), "global_win_rate": p_bar,
               "n_directional": total, "conf": conf}
    with open(config.INDICATOR_CONF_PATH, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info("direct-conf: fitted for %s (global p=%.3f, n=%d)",
                sorted(conf), p_bar, total)
    return payload


def run_nightly_training(store) -> Dict[str, Any]:
    rows = store.training_rows()
    meta = train_meta_model(rows)
    calib = fit_calibration(rows)
    direct_conf = fit_direct_conf(rows)

    # Ecosystem refresh (B4): best-effort — network failures keep the current
    # (cached or hardcoded) lists; reload applies the new cache in-process.
    ecosystems_refreshed = False
    if config.ECOSYSTEMS_AUTO:
        try:
            from scripts.refresh_ecosystems import refresh
            refresh()
            from agents.research_agent import load_ecosystems_cache
            ecosystems_refreshed = load_ecosystems_cache()
        except Exception as e:
            logger.error("ecosystem refresh failed: %s", e)

    return {"rows": len(rows), "meta": meta,
            "calibrated_tfs": sorted((calib or {}).get("knots", {})),
            "direct_conf_indicators": sorted((direct_conf or {}).get("conf", {})),
            "ecosystems_refreshed": ecosystems_refreshed}


# --------------------------------------------------------------------------- #
# Inference (hot path — cached artifacts, sklearn only if a model exists)
# --------------------------------------------------------------------------- #
def _load_calibration() -> Optional[Dict[str, Any]]:
    path = config.CALIBRATION_PATH
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return None
    if _CALIB_CACHE["path_mtime"] != mtime:
        try:
            with open(path) as f:
                _CALIB_CACHE["knots"] = json.load(f).get("knots", {})
            _CALIB_CACHE["path_mtime"] = mtime
        except Exception:
            return None
    return _CALIB_CACHE["knots"]


def apply_calibration(tf: str, conf: float) -> Optional[float]:
    """Calibrated confidence via np.interp on the exported knots; None when no
    calibration exists for this TF (callers show the raw conf only)."""
    knots = _load_calibration()
    if not knots or tf not in knots:
        return None
    k = knots[tf]
    return float(np.interp(conf, k["x"], k["y"]))


def meta_probability(row_like: Dict[str, Any]) -> Optional[float]:
    """p(correct) for a prediction-shaped dict; None when no model artifact."""
    path = config.META_MODEL_PATH
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return None
    if _MODEL_CACHE["path_mtime"] != mtime:
        try:
            import joblib
            _MODEL_CACHE["model"] = joblib.load(path)["model"]
            _MODEL_CACHE["path_mtime"] = mtime
        except Exception:
            return None
    model = _MODEL_CACHE["model"]
    if model is None:
        return None
    try:
        x = np.array([meta_features_from_prediction_row(row_like)], dtype=float)
        return float(model.predict_proba(x)[0, 1])
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Background loop
# --------------------------------------------------------------------------- #
async def nightly_loop(application, hour_ist: Optional[int] = None) -> None:
    hour = hour_ist if hour_ist is not None else config.NIGHTLY_HOUR_IST
    bd = application.bot_data
    logger.info("nightly trainer started (%02d:00 IST)", hour)
    while True:
        try:
            now = datetime.now(tz=IST)
            nxt = now.replace(hour=hour, minute=0, second=0, microsecond=0)
            if nxt <= now:
                nxt += timedelta(days=1)
            await asyncio.sleep((nxt - now).total_seconds())

            summary = await asyncio.to_thread(run_nightly_training, bd["store"])
            logger.info("nightly training: %s", summary)
            dev_chat = getattr(bd.get("broadcaster"), "dev_chat_id", None)
            if dev_chat:
                m = summary.get("meta") or {}
                txt = (f"🌙 <b>nightly training</b>\nrows={summary['rows']} "
                       f"calibrated={summary['calibrated_tfs']}\n"
                       f"meta: AUC={m.get('holdout_auc')} lift={m.get('precision_lift')} "
                       f"(n_hold={m.get('n_holdout')})"
                       if m else
                       f"🌙 <b>nightly training</b>\nrows={summary['rows']} — "
                       f"below META_MIN_ROWS, model skipped; "
                       f"calibrated={summary['calibrated_tfs']}")
                try:
                    await application.bot.send_message(chat_id=dev_chat, text=txt,
                                                       parse_mode="HTML")
                except Exception as e:
                    logger.debug("nightly summary send failed: %s", e)
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error("nightly error: %s", e)
            await asyncio.sleep(3600)
