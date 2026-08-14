"""Evidence ledger (v3.8) — per-cohort empirical precision for every emission
candidate source, with Wilson lower bounds. The edge-first gate emits a
candidate only when its cohort has EARNED it (or the source is in a bounded
probation), replacing the v3.5-v3.7 per-source hand flags whose tiny-sample
verdicts the 21-day audit refuted (GATE_1H_MIXED blocked 96 crossings @40.6%).

Cohort key: source | tf | regime_group | vol_band
  source        candidate_trigger vocabulary: nwe, sms, sms_bos, sms_choch,
                trend, conf (exact string, ledger prices sms variants apart)
  regime_group  trending (trend_up/trend_down) | ranging | mixed
  vol_band      calm (<0.3) | normal | elevated (>=0.7) | unknown

Correctness = candidate_action == realized_label, i.e. the direction the
candidate itself proposed — NOT the brain final (grade-what-was-sent).

Artifact: logs/emission_ledger.json — rebuilt by the nightly job from
training rows and seeded once at deploy by scripts/seed_ledger.py.
"""
from __future__ import annotations

import json
import logging
import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import config

logger = logging.getLogger("bitreinforcex.ledger")

_Z = 1.96


def wilson_lb(successes: int, n: int, z: float = _Z) -> float:
    """Lower bound of the Wilson score interval (0 when n == 0)."""
    if n <= 0:
        return 0.0
    p = successes / n
    denom = 1.0 + z * z / n
    center = p + z * z / (2 * n)
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)
    return max(0.0, (center - margin) / denom)


def regime_group(regime: Optional[str]) -> str:
    r = (regime or "").lower()
    if r in ("trend_up", "trend_down"):
        return "trending"
    if r == "ranging":
        return "ranging"
    return "mixed"


def vol_band(vol_pct: Optional[float]) -> str:
    if vol_pct is None:
        return "unknown"
    try:
        v = float(vol_pct)
    except (TypeError, ValueError):
        return "unknown"
    if v < 0.3:
        return "calm"
    if v >= 0.7:
        return "elevated"
    return "normal"


def _key(source: str, tf: str, rg: str, vb: str) -> str:
    return "|".join((source, tf, rg, vb))


def synthesize_candidate(r: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    """(source, action) for a LEGACY row (pre-v3.8: candidate columns NULL),
    reconstructed conservatively from the v3.7.1 funnel telemetry. Lives here
    — inside the builder's own code path — so the nightly rebuild produces
    the SAME cohorts as the deploy-time seed; when it lived only in the seed
    script, the first nightly overwrote the seeded ledger with empty cohorts
    and every source fell back to new-source probation.

      nwe_* / no_brain_agreement                  -> ("nwe", nwe_action)
      low_volume with a directional nwe_action    -> ("nwe", nwe_action)
      meta_gate / conf_saturated (checked BEFORE the conf prefix — the final
        suppressor overwrote the reason): nwe_action first, else conf when
        final_confidence cleared the gate
      conf_* / counter_trend_conf                 -> ("conf", final_action)
      trend rows: only EMITTED ones, direction approximated by the brain
        final (the sent trend_action was not persisted pre-v3.8 — a known
        approximation; suppressed trend candidates are skipped, not guessed)
    """
    reason = (r.get("gate_reason") or "").lower()
    nwe = (r.get("nwe_action") or "").lower()
    final = (r.get("final_action") or "").lower()
    conf = float(r.get("final_confidence") or 0.0)
    src = act = None
    if reason.startswith("nwe") or reason == "no_brain_agreement":
        src, act = "nwe", nwe
    elif reason == "low_volume" and nwe in ("buy", "sell"):
        src, act = "nwe", nwe
    elif reason in ("meta_gate", "conf_saturated"):
        if nwe in ("buy", "sell"):
            src, act = "nwe", nwe
        elif final in ("buy", "sell") and conf >= config.CONFIDENCE_GATE:
            src, act = "conf", final
    elif reason.startswith("conf") or reason == "counter_trend_conf":
        src, act = "conf", final
    elif (reason.startswith(("trend", "reversal", "counter_trend"))
          and r.get("emitted") and final in ("buy", "sell")):
        src, act = "trend", final
    if src and act in ("buy", "sell"):
        return src, act
    return None, None


def build_ledger(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate graded candidate rows into the cohort table.

    ``rows`` are Store.training_rows() dicts (prediction JOIN outcome). v3.8
    rows carry candidate_trigger/candidate_action natively; legacy rows are
    synthesized (see synthesize_candidate) so seed and nightly rebuilds are
    one code path.
    """
    cohorts: Dict[str, Dict[str, int]] = {}
    emitted_by_source: Dict[str, int] = {}
    for r in rows:
        src = (r.get("candidate_trigger") or "").lower()
        act = (r.get("candidate_action") or "").lower()
        if not src or act not in ("buy", "sell"):
            s2, a2 = synthesize_candidate(r)
            if not s2:
                continue
            src, act = s2, a2
        label = (r.get("realized_label") or "").lower()
        tf = (r.get("tf") or "").lower()
        rf = r.get("regime_feats") or {}
        rg = regime_group(r.get("regime") or rf.get("regime"))
        vb = vol_band(rf.get("vol_pct"))
        hit = int(act == label)
        for key in (_key(src, tf, rg, vb), _key(src, tf, "*", "*")):
            c = cohorts.setdefault(key, {"n": 0, "hit": 0})
            c["n"] += 1
            c["hit"] += hit
        if r.get("emitted"):
            emitted_by_source[src] = emitted_by_source.get(src, 0) + 1
    for c in cohorts.values():
        c["lb"] = round(wilson_lb(c["hit"], c["n"]), 4)
        c["rate"] = round(c["hit"] / c["n"], 4) if c["n"] else 0.0
    return {"built_ts": time.time(), "cohorts": cohorts,
            "emitted_by_source": emitted_by_source}


def save_ledger(ledger: Dict[str, Any], path: Optional[str] = None) -> None:
    path = path or config.LEDGER_PATH
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(ledger, f, indent=1)


_CACHE: Dict[str, Any] = {"path_mtime": None, "ledger": None}


def load_ledger_cached(path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """mtime-cached artifact load (same contract as the meta-model cache);
    None when the artifact does not exist — the gate then emits nothing but
    probation, and preflight warns loudly."""
    path = path or config.LEDGER_PATH
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return None
    if _CACHE["path_mtime"] != mtime:
        try:
            with open(path, "r", encoding="utf-8") as f:
                _CACHE["ledger"] = json.load(f)
            _CACHE["path_mtime"] = mtime
        except Exception:
            return None
    return _CACHE["ledger"]


def ledger_verdict(ledger: Optional[Dict[str, Any]], source: str, tf: str,
                   regime: Optional[str], vol_pct: Optional[float]
                   ) -> Tuple[bool, str, Dict[str, Any]]:
    """(eligible, reason, stats) for one candidate under the strict posture.

    The strict test, two parts (a pure LB floor emits nothing at day-21
    sample sizes): measured rate >= LEDGER_FLOOR AND Wilson LB >=
    LEDGER_LB_GUARD (the anti-fluke bound: pessimistically still above
    random). Resolution order:

    1. cohort with n >= LEDGER_MIN_N        -> its own test
    2. else source-global with n >= MIN_N   -> global test ("probation via
       global": proven sources may enter unmeasured cohorts)
    3. else brand-new source                -> bounded probation while its
       lifetime emitted count < LEDGER_PROBATION_N
    """
    def _passes(c: Dict[str, Any]) -> bool:
        return (c["rate"] >= config.LEDGER_FLOOR
                and c["lb"] >= config.LEDGER_LB_GUARD)

    src = (source or "").lower()
    stats: Dict[str, Any] = {"source": src}
    if not ledger:
        return False, "ledger_missing", stats
    cohorts = ledger.get("cohorts") or {}
    ck = _key(src, tf, regime_group(regime), vol_band(vol_pct))
    cohort = cohorts.get(ck)
    if cohort and cohort["n"] >= config.LEDGER_MIN_N:
        stats.update({"cohort": ck, "n": cohort["n"], "lb": cohort["lb"],
                      "rate": cohort["rate"]})
        if _passes(cohort):
            return True, "ledger_ok", stats
        return False, "ledger_below_floor", stats
    gk = _key(src, tf, "*", "*")
    glob = cohorts.get(gk)
    if glob and glob["n"] >= config.LEDGER_MIN_N:
        stats.update({"cohort": gk, "n": glob["n"], "lb": glob["lb"],
                      "rate": glob["rate"], "probation": "global"})
        if _passes(glob):
            return True, "ledger_ok", stats
        return False, "ledger_below_floor", stats
    # Provisional tier: a source with SOME evidence (n in [MIN_N/3, MIN_N))
    # is judged on its rate alone — it must never slide back into the
    # new-source probation budget (a 5.9%-hit trend source at n=17 would
    # otherwise get a fresh 40-emission allowance). Suppressed candidates
    # keep getting graded, so a good source revives at n >= MIN_N.
    prov_n = max(8, config.LEDGER_MIN_N // 3)
    if glob and glob["n"] >= prov_n:
        stats.update({"cohort": gk, "n": glob["n"], "lb": glob["lb"],
                      "rate": glob["rate"], "probation": "provisional"})
        if glob["rate"] >= config.LEDGER_FLOOR:
            return True, "ledger_ok", stats
        return False, "ledger_below_floor", stats
    emitted = (ledger.get("emitted_by_source") or {}).get(src, 0)
    stats.update({"cohort": None, "n": (glob or {}).get("n", 0),
                  "probation": "new_source", "emitted_so_far": emitted})
    if emitted < config.LEDGER_PROBATION_N:
        return True, "ledger_probation", stats
    return False, "ledger_cold", stats
