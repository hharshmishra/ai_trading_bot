"""Backtest metrics: per-group precision/expectancy + A/B comparison.

Pure numpy/python — no scipy. Groups are keyed ``tf|regime|reason`` so the
regime dimension appears automatically once Phase 2 starts stamping it
(``regime=None`` groups under "all").
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional


def _wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, centre - half), min(1.0, centre + half))


def _group_key(e: Dict[str, Any]) -> str:
    return f"{e.get('tf')}|{e.get('regime') or 'all'}|{e.get('reason')}"


def summarize(emissions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate labeled emissions into per-group and total metrics.

    - tb_precision: TP-first rate among barrier-decided emissions (tp+sl)
    - expectancy_r: mean R multiple (tp -> +tp_mult, sl -> -sl_mult,
      timeout -> price move / ATR); "incomplete" rows are excluded
    - fixed_hit_rate: legacy fixed-horizon label == predicted action
    """
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for e in emissions:
        groups.setdefault(_group_key(e), []).append(e)

    out_groups: Dict[str, Any] = {}
    for key, rows in sorted(groups.items()):
        tp = sum(1 for r in rows if r["label_tb"] == "tp")
        sl = sum(1 for r in rows if r["label_tb"] == "sl")
        to = sum(1 for r in rows if r["label_tb"] == "timeout")
        inc = sum(1 for r in rows if r["label_tb"] == "incomplete")
        decided = tp + sl
        graded = decided + to

        rs = []
        for r in rows:
            if r["label_tb"] == "tp":
                rs.append(float(r["tp_mult"]))
            elif r["label_tb"] == "sl":
                rs.append(-float(r["sl_mult"]))
            elif r["label_tb"] == "timeout" and r.get("atr") and r.get("fwd_return") is not None:
                move = float(r["fwd_return"]) * float(r["entry"])
                signed = move if r["action"] == "buy" else -move
                rs.append(signed / float(r["atr"]))

        fixed_known = [r for r in rows if r.get("label_fixed") is not None]
        fixed_hits = sum(1 for r in fixed_known if r["label_fixed"] == r["action"])

        lo, hi = _wilson_ci(tp, decided)
        out_groups[key] = {
            "n": len(rows),
            "tp": tp, "sl": sl, "timeout": to, "incomplete": inc,
            "tb_precision": (tp / decided) if decided else None,
            "tb_precision_ci": [lo, hi] if decided else None,
            "timeout_share": (to / graded) if graded else None,
            "expectancy_r": (sum(rs) / len(rs)) if rs else None,
            "fixed_hit_rate": (fixed_hits / len(fixed_known)) if fixed_known else None,
            "ambiguous": sum(1 for r in rows if r.get("ambiguous")),
        }

    return {
        "total_emissions": len(emissions),
        "groups": out_groups,
    }


def _two_proportion_z(p1: float, n1: int, p2: float, n2: int) -> Optional[float]:
    """z statistic for H0: p1 == p2 (pooled). None when undefined."""
    if not n1 or not n2:
        return None
    pooled = (p1 * n1 + p2 * n2) / (n1 + n2)
    var = pooled * (1 - pooled) * (1 / n1 + 1 / n2)
    if var <= 0:
        return None
    return (p2 - p1) / math.sqrt(var)


def compare(baseline: Dict[str, Any], candidate: Dict[str, Any]) -> Dict[str, Any]:
    """Per-group deltas candidate - baseline, with a two-proportion z on
    tb_precision (|z| >= 1.96 ~ 95% significance)."""
    out: Dict[str, Any] = {"groups": {}}
    b_groups = baseline.get("groups", {})
    c_groups = candidate.get("groups", {})
    for key in sorted(set(b_groups) | set(c_groups)):
        b, c = b_groups.get(key), c_groups.get(key)
        row: Dict[str, Any] = {"in_baseline": b is not None, "in_candidate": c is not None}
        if b and c and b.get("tb_precision") is not None and c.get("tb_precision") is not None:
            nb, nc = b["tp"] + b["sl"], c["tp"] + c["sl"]
            row["tb_precision_delta"] = c["tb_precision"] - b["tb_precision"]
            row["z"] = _two_proportion_z(b["tb_precision"], nb, c["tb_precision"], nc)
            row["significant_95"] = bool(row["z"] is not None and abs(row["z"]) >= 1.96)
            if b.get("expectancy_r") is not None and c.get("expectancy_r") is not None:
                row["expectancy_r_delta"] = c["expectancy_r"] - b["expectancy_r"]
            row["n_delta"] = c["n"] - b["n"]
        out["groups"][key] = row
    return out
