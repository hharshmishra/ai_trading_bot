#!/usr/bin/env python3
"""SMS event backtest (v3.8) — the pre-registered SMS_EMIT gate.

Replays the ported Smart Money Structure sources (sms / sms_bos / sms_choch)
over the cached OHLCV history using the SAME labeling pipeline production
grading uses (ATR barriers -> triple_barrier -> tp/sl/timeout -> label), and
compares each (source, tf, side) cohort's precision against the per-TF
directional base rate.

Decision rule (pre-registered in the v3.8 plan BEFORE this ran):
    SMS_EMIT=true  iff  pooled event precision Wilson-LB > base + 5pts
                        on >= 100 events for at least one (tf x side) cohort
    else SMS ships shadow-only (recorded + graded, never emitted).

Vectorized full-series evaluation is valid because sms_structure is causal
with bounded lookback (parity test: tests/test_sms_indicator.py).

Usage: venv/bin/python scripts/backtest_sms.py [--pairs N] [--out DIR]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
os.environ.setdefault("BITREINFORCEX_NO_DOTENV", "1")

import pandas as pd  # noqa: E402

from agents import custom_indicators as ci  # noqa: E402
from agents.regime_agent import classify_regime  # noqa: E402
from backtest.metrics import _wilson_ci  # noqa: E402
from grader import HORIZON_K, THRESHOLD  # noqa: E402
from grading.barriers import atr_from_ohlcv, barrier_prices, triple_barrier  # noqa: E402

HIST = os.path.join(ROOT, "data", "history")
WARMUP = 500
TFS = ("1h", "4h")
TP_SL = {"1h": (1.5, 1.0), "4h": (1.5, 1.0)}

EVENT_COLS = (("sms_buy", "sms", "buy"), ("sms_sell", "sms", "sell"),
              ("sms_bos_buy", "sms_bos", "buy"), ("sms_bos_sell", "sms_bos", "sell"),
              ("sms_choch_buy", "sms_choch", "buy"), ("sms_choch_sell", "sms_choch", "sell"))


def _fixed_label(fr: float, tf: str) -> str:
    th = THRESHOLD[tf]
    return "buy" if fr >= th else ("sell" if fr <= -th else "skip")


def _vol_band(v) -> str:
    if v is None:
        return "unknown"
    return "calm" if v < 0.3 else ("elevated" if v >= 0.7 else "normal")


def _label_event(df: pd.DataFrame, t: int, side: str, tf: str) -> dict | None:
    """Production-identical labeling for an event at bar t."""
    k = HORIZON_K[tf]
    n = len(df)
    if t + k >= n:
        return None
    entry = float(df["close"].iloc[t])
    wdf = df.iloc[max(0, t - WARMUP + 1): t + 1]
    atr = atr_from_ohlcv(wdf)
    label_tb = None
    if atr:
        tp_m, sl_m = TP_SL[tf]
        tp, sl = barrier_prices(entry, atr, side, tp_m, sl_m)
        out = triple_barrier(df.iloc[t + 1: t + 1 + k], entry, side, tp, sl, k)
        if out.label_tb != "incomplete":
            label_tb = out.label_tb
    fr = (float(df["close"].iloc[t + k]) - entry) / entry
    if label_tb in ("tp", "sl"):
        label = side if label_tb == "tp" else ("sell" if side == "buy" else "buy")
    else:
        label = _fixed_label(fr, tf)
    snap = classify_regime(wdf)
    return {"side": side, "label": label, "label_tb": label_tb,
            "regime": snap.regime, "vol_band": _vol_band(snap.vol_pct)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=12)
    ap.add_argument("--out", default=os.path.join(ROOT, "logs", "backtest", "sms-v38"))
    args = ap.parse_args()

    pairs = sorted({f.split("_")[0] for f in os.listdir(HIST) if f.endswith(".csv")})
    pairs = [p for p in pairs if all(
        os.path.exists(os.path.join(HIST, f"{p}_{tf}.csv")) for tf in TFS)][: args.pairs]

    events: list[dict] = []
    base_counts: dict[str, dict[str, int]] = {tf: {"dir": 0, "n": 0} for tf in TFS}
    for pair in pairs:
        for tf in TFS:
            df = pd.read_csv(os.path.join(HIST, f"{pair}_{tf}.csv"))
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            d = ci.sms_structure(df)
            if d is None:
                continue
            k = HORIZON_K[tf]
            # per-TF directional base rate: P(random side correct) on the
            # fixed-horizon label, sampled on every bar past warmup
            closes = df["close"].astype(float).to_numpy()
            for t in range(WARMUP, len(df) - k, 7):        # stride: iid-enough
                fr = (closes[t + k] - closes[t]) / closes[t]
                lab = _fixed_label(fr, tf)
                base_counts[tf]["n"] += 1
                base_counts[tf]["dir"] += int(lab in ("buy", "sell"))
            for col, source, side in EVENT_COLS:
                for t in d.index[d[col]].tolist():
                    if t < WARMUP:
                        continue
                    row = _label_event(df, int(t), side, tf)
                    if row:
                        row.update({"pair": pair, "tf": tf, "source": source})
                        events.append(row)
        print(f"{pair}: {len(events)} events cumulative", flush=True)

    base = {tf: (0.5 * c["dir"] / c["n"] if c["n"] else 0.0)
            for tf, c in base_counts.items()}

    def _agg(rows):
        n = len(rows)
        hit = sum(r["label"] == r["side"] for r in rows)
        lb, ub = _wilson_ci(hit, n) if n else (0.0, 0.0)
        return {"n": n, "hit": hit, "rate": round(hit / n, 4) if n else None,
                "wilson_lb": round(lb, 4), "wilson_ub": round(ub, 4)}

    cohorts = {}
    for r in events:
        for key in ((r["source"], r["tf"], r["side"]),
                    (r["source"], r["tf"], "all"),
                    (r["source"], r["tf"], r["side"], r["regime"], r["vol_band"])):
            cohorts.setdefault("|".join(key), []).append(r)
    summary = {k: _agg(v) for k, v in sorted(cohorts.items())}

    # pre-registered decision
    verdict, winners = False, []
    for (source, tf, side) in {(r["source"], r["tf"], r["side"]) for r in events}:
        agg = summary["|".join((source, tf, side))]
        if agg["n"] >= 100 and agg["wilson_lb"] > base[tf] + 0.05:
            verdict = True
            winners.append({"source": source, "tf": tf, "side": side, **agg,
                            "base": round(base[tf], 4)})

    os.makedirs(args.out, exist_ok=True)
    report = {"pairs": pairs, "tfs": list(TFS), "warmup": WARMUP,
              "n_events": len(events), "base_rate_random_side": {
                  tf: round(b, 4) for tf, b in base.items()},
              "decision_rule": "SMS_EMIT=true iff wilson_lb > base+5pts on n>=100 for any (source,tf,side)",
              "sms_emit": verdict, "winning_cohorts": winners,
              "cohorts": summary}
    with open(os.path.join(args.out, "report.json"), "w") as f:
        json.dump(report, f, indent=2)

    lines = ["# SMS backtest (v3.8)\n",
             f"pairs: {len(pairs)}  events: {len(events)}  "
             f"base(random side): " + ", ".join(f"{tf}={base[tf]:.1%}" for tf in TFS),
             f"\n**verdict: SMS_EMIT={'true' if verdict else 'false'}**\n",
             "| cohort | n | hit rate | wilson LB |", "|---|---|---|---|"]
    for key, agg in summary.items():
        if key.count("|") == 2 and agg["n"] >= 20:
            lines.append(f"| {key} | {agg['n']} | {agg['rate']:.1%} | {agg['wilson_lb']:.1%} |")
    with open(os.path.join(args.out, "report.md"), "w") as f:
        f.write("\n".join(lines) + "\n")

    print(json.dumps({"sms_emit": verdict, "n_events": len(events),
                      "base": base, "winners": winners}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
