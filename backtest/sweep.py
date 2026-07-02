"""Walk-forward parameter sweep.

Grid combos are selected on the TRAIN slice (first ``frac`` of history) only;
the winner is then evaluated once on the untouched TEST slice. Selecting on
train and confirming on test is the guard against curve-fitting — a combo that
wins train but collapses on test is rejected in favour of production defaults.

Sweepable today: NWE kernel (nwe_h, nwe_mult) and barrier widths
(tp_mult, sl_mult). Phase 2 adds regime thresholds via config env overrides.
"""
from __future__ import annotations

import itertools
from typing import Any, Dict, List, Optional

import pandas as pd

from backtest.engine import replay_pair
from backtest.metrics import summarize


def grid_from_spec(spec: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """{"nwe_h": [6, 8, 10], "tp_mult": [1.5, 2.0]} -> list of combo dicts."""
    keys = sorted(spec)
    return [dict(zip(keys, vals)) for vals in itertools.product(*(spec[k] for k in keys))]


def walk_forward_split(df: pd.DataFrame, frac: float = 0.7):
    cut = int(len(df) * frac)
    return df.iloc[:cut].reset_index(drop=True), df.iloc[cut:].reset_index(drop=True)


def _overall_objective(summary: Dict[str, Any], objective: str) -> Optional[float]:
    """Weighted (by n) mean of the per-group objective."""
    total_n, acc = 0, 0.0
    for g in summary.get("groups", {}).values():
        v = g.get(objective)
        if v is None:
            continue
        acc += v * g["n"]
        total_n += g["n"]
    return (acc / total_n) if total_n else None


def run_combo(history: Dict[str, pd.DataFrame], tf: str, combo: Dict[str, Any],
              *, k: int, theta: float, window: int = 500,
              agent_factory=None) -> Dict[str, Any]:
    from agents.indicator_agent import IndicatorAgent
    factory = agent_factory or IndicatorAgent
    agent = factory(nwe_h=combo.get("nwe_h", 8.0), nwe_mult=combo.get("nwe_mult", 3.0))
    emissions: List[Dict[str, Any]] = []
    for pair, df in history.items():
        r = replay_pair(df, pair, tf, agent=agent, k=k, theta=theta, window=window,
                        tp_mult=combo.get("tp_mult"), sl_mult=combo.get("sl_mult"))
        emissions.extend(r.emissions)
    return summarize(emissions)


def sweep(history: Dict[str, pd.DataFrame], tf: str, grid: List[Dict[str, Any]],
          *, k: int, theta: float, window: int = 500, frac: float = 0.7,
          objective: str = "tb_precision", min_emissions: int = 30,
          agent_factory=None) -> Dict[str, Any]:
    """Returns {"rows": [...], "best": {...}} — best chosen on train, reported
    with its (single-shot) test evaluation."""
    train = {p: walk_forward_split(df, frac)[0] for p, df in history.items()}
    test = {p: walk_forward_split(df, frac)[1] for p, df in history.items()}

    rows = []
    for combo in grid:
        s = run_combo(train, tf, combo, k=k, theta=theta, window=window,
                      agent_factory=agent_factory)
        score = _overall_objective(s, objective)
        rows.append({"params": combo, "train_score": score,
                     "train_emissions": s["total_emissions"]})

    eligible = [r for r in rows if r["train_score"] is not None
                and r["train_emissions"] >= min_emissions]
    if not eligible:
        return {"rows": rows, "best": None}

    best = max(eligible, key=lambda r: r["train_score"])
    # single-shot confirmation on the untouched tail
    s_test = run_combo(test, tf, best["params"], k=k, theta=theta, window=window,
                       agent_factory=agent_factory)
    best = dict(best)
    best["test_score"] = _overall_objective(s_test, objective)
    best["test_emissions"] = s_test["total_emissions"]
    best["holds_on_test"] = (best["test_score"] is not None
                             and best["train_score"] is not None
                             and best["test_score"] >= 0.9 * best["train_score"])
    return {"rows": rows, "best": best}
