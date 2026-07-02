"""Meta-label feature builder — ONE function used at train time (DB rows) and
decide time (live rows). A single code path is the guard against train/serve
skew: if the live vector were built differently from the training vector, the
model's probabilities would be silently meaningless.

Input: a prediction-row-shaped dict. At train time that is a
``Store.training_rows()`` row; at decide time ``cycle`` builds the same shape
from the in-flight decision BEFORE any outcome exists — so every feature here
must be knowable at prediction time (never touch outcome fields).
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

REGIMES = ("trend_up", "trend_down", "ranging", "mixed")
TFS = ("1h", "4h", "1d", "1w")
TRIGGER_GROUPS = ("nwe", "trend", "conf")   # + none -> all zeros

FEATURE_NAMES: List[str] = (
    [f"regime_{r}" for r in REGIMES]
    + ["adx", "chop", "vol_pct", "atr_pct", "vol_ok"]
    + ["agreement", "n_voters"]
    + ["final_conf", "indicator_conf", "research_conf", "news_conf", "deriv_conf"]
    + ["deriv_available", "funding_extreme", "funding_z_abs"]
    + [f"tf_{t}" for t in TFS]
    + ["hour_sin", "hour_cos"]
    + [f"trigger_{g}" for g in TRIGGER_GROUPS]
    + ["emitted"]
)


def _f(v: Any, default: float = 0.0) -> float:
    try:
        x = float(v)
        return x if math.isfinite(x) else default
    except (TypeError, ValueError):
        return default


def _trigger_group(trigger: Optional[str]) -> Optional[str]:
    t = (trigger or "").lower()
    if t.startswith("nwe"):
        return "nwe"
    if t.startswith("trend"):
        return "trend"
    if t.startswith("conf"):
        return "conf"
    return None


def meta_features_from_prediction_row(p: Dict[str, Any]) -> List[float]:
    """Fixed-length vector aligned with FEATURE_NAMES."""
    rf = p.get("regime_feats") or {}
    regime = p.get("regime") or rf.get("regime") or "mixed"
    feats: List[float] = [1.0 if regime == r else 0.0 for r in REGIMES]

    entry = _f(p.get("entry_price"))
    atr = _f(p.get("atr") if p.get("atr") is not None else rf.get("atr"))
    feats += [
        min(_f(rf.get("adx")) / 50.0, 1.0),
        min(_f(rf.get("chop")) / 100.0, 1.0),
        _f(rf.get("vol_pct"), 0.5),
        min(atr / entry, 0.25) * 4.0 if entry > 0 and atr > 0 else 0.0,
        1.0 if rf.get("vol_ok", True) else 0.0,
    ]

    final_action = (p.get("final_action") or "skip").lower()
    votes, agree = 0, 0
    for prefix in ("indicator", "research", "news", "deriv"):
        a = p.get(f"{prefix}_action")
        if a is None:
            continue
        votes += 1
        if str(a).lower() == final_action:
            agree += 1
    feats += [agree / votes if votes else 0.0, votes / 4.0]

    feats += [
        _f(p.get("final_confidence")),
        _f(p.get("indicator_conf")),
        _f(p.get("research_conf")),
        _f(p.get("news_conf")),
        _f(p.get("deriv_conf")),
    ]

    dfeats = p.get("deriv_feats")
    if isinstance(dfeats, list) and len(dfeats) >= 3:
        feats += [1.0, _f(dfeats[2]), abs(_f(dfeats[1]))]
    else:
        feats += [0.0, 0.0, 0.0]

    tf = (p.get("tf") or p.get("timeframe") or "").lower()
    feats += [1.0 if tf == t else 0.0 for t in TFS]

    ts = p.get("candle_close_ts") or p.get("created_ts") or 0.0
    hour = (float(ts) % 86400.0) / 3600.0
    feats += [math.sin(2 * math.pi * hour / 24.0), math.cos(2 * math.pi * hour / 24.0)]

    group = _trigger_group(p.get("trigger_source"))
    feats += [1.0 if group == g else 0.0 for g in TRIGGER_GROUPS]
    feats += [1.0 if p.get("emitted") else 0.0]

    assert len(feats) == len(FEATURE_NAMES)
    return feats
