#!/usr/bin/env python
"""Indicator redundancy + standalone win-rate report (enhancement D3).

Report-only — wires into NOTHING. Over the cached backtest history
(data/history/*.csv), computes each indicator's per-bar directional signal
series, then:
  1. Spearman correlation matrix (|rho| > 0.75 pairs flagged redundant)
  2. per-indicator standalone triple-barrier win rates (production barriers)

Output: logs/backtest/indicator-corr.md — informs human weighting/pruning
decisions; production behavior is untouched by this script.

    python scripts/analyze_indicators.py --pairs BTCUSDT,ETHUSDT,SOLUSDT --tf 4h
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

import config
from grading.barriers import barrier_prices, triple_barrier
from grader import HORIZON_K


def signal_series(df: pd.DataFrame) -> pd.DataFrame:
    """Per-bar directional readings (+1/-1/0) for every indicator family."""
    import pandas_ta as ta
    from agents import custom_indicators as ci

    out = pd.DataFrame(index=df.index)
    close, high, low, vol = df["close"], df["high"], df["low"], df["volume"]

    nwe = ci.apply_nadaraya_watson_envelope(df.copy())
    out["nwe_state"] = np.where(nwe["close"] < nwe["nwe_lower"], 1,
                                np.where(nwe["close"] > nwe["nwe_upper"], -1, 0))
    st = ci.supertrend_fast(high, low, close, length=10, multiplier=3)
    out["supertrend_dir"] = st["SUPERTd_10_3.0"]
    out["macd_sign"] = np.sign(ta.macd(close)["MACDh_12_26_9"]).fillna(0)
    rsi = ta.rsi(close, length=14)
    out["rsi_zone"] = np.where(rsi < 30, 1, np.where(rsi > 70, -1, 0))
    bb = ta.bbands(close, length=20, std=2)
    out["bb_pos"] = np.where(close <= bb["BBL_20_2.0"], 1,
                             np.where(close >= bb["BBU_20_2.0"], -1, 0))
    ma20 = close.rolling(20).mean()
    ma50 = close.rolling(50).mean()
    out["ma_ribbon"] = np.where((close > ma20) & (close > ma50), 1,
                                np.where((close < ma20) & (close < ma50), -1, 0))
    ce = ci.chandelier_exit(df.copy())
    out["chandelier"] = ce["ce_signal"].map({"buy": 1, "sell": -1}).fillna(0)
    at = ci.alpha_trend(df.copy())
    out["alpha_trend"] = at["alpha_signal"].map({"buy": 1, "sell": -1}).fillna(0)
    # divergences (D1/D2) — rolling last-bar evaluation is expensive; sample
    # every 4th bar for the correlation study
    rsi_div, obv_div = np.zeros(len(df)), np.zeros(len(df))
    obv = ta.obv(close, vol)
    for i in range(80, len(df), 4):
        rsi_div[i] = ci.pivot_divergence(close.iloc[:i + 1], rsi.iloc[:i + 1])
        obv_div[i] = ci.pivot_divergence(close.iloc[:i + 1], obv.iloc[:i + 1])
    out["rsi_div"] = rsi_div
    out["obv_div"] = obv_div
    return out.fillna(0)


def standalone_winrates(df: pd.DataFrame, sig: pd.DataFrame, tf: str) -> dict:
    """First-touch TB outcome for each bar where an indicator reads ±1."""
    from grading.barriers import atr_from_ohlcv
    k = HORIZON_K.get(tf, 2)
    tp_mult, sl_mult = config.BARRIER_MULTS.get(tf, (1.5, 1.0))
    res = {}
    closes = df["close"].to_numpy(dtype=float)
    for col in sig.columns:
        s = sig[col].to_numpy()
        wins = losses = 0
        idxs = np.nonzero(s[:-k - 1])[0]
        idxs = idxs[idxs > 60]
        if len(idxs) > 400:                       # cap for runtime
            idxs = idxs[:: max(1, len(idxs) // 400)]
        for t in idxs:
            direction = "buy" if s[t] > 0 else "sell"
            window = df.iloc[max(0, t - 60): t + 1]
            atr = atr_from_ohlcv(window, config.ATR_LEN)
            if not atr:
                continue
            tp, sl = barrier_prices(closes[t], atr, direction, tp_mult, sl_mult)
            out = triple_barrier(df.iloc[t + 1: t + 1 + k], closes[t], direction, tp, sl, k)
            if out.label_tb == "tp":
                wins += 1
            elif out.label_tb == "sl":
                losses += 1
        n = wins + losses
        res[col] = {"n_decided": n, "win_rate": round(wins / n, 3) if n else None}
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", default="BTCUSDT,ETHUSDT,SOLUSDT,ADAUSDT")
    ap.add_argument("--tf", default="4h")
    ap.add_argument("--out", default="logs/backtest/indicator-corr.md")
    args = ap.parse_args()

    pairs = [p.strip().upper() for p in args.pairs.split(",")]
    all_sig, all_win = [], {}
    for pair in pairs:
        path = f"data/history/{pair}_{args.tf}.csv"
        if not os.path.exists(path):
            print(f"  {pair}: no cached history ({path}) — run a backtest first")
            continue
        df = pd.read_csv(path)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", errors="coerce")
        sig = signal_series(df)
        all_sig.append(sig)
        for k_, v in standalone_winrates(df, sig, args.tf).items():
            all_win.setdefault(k_, []).append(v)
        print(f"  {pair}: {len(df)} bars analysed")

    if not all_sig:
        print("no data — aborting")
        return 1

    corr = pd.concat(all_sig).corr(method="spearman").round(2)
    lines = [f"# Indicator redundancy & standalone win rates — {args.tf}, {len(all_sig)} pairs", ""]
    lines.append("## Spearman correlation (|rho| > 0.75 = redundant pair)")
    lines.append("")
    lines.append(corr.to_markdown())
    lines.append("")
    flagged = [(a, b, corr.loc[a, b]) for a in corr.index for b in corr.columns
               if a < b and abs(corr.loc[a, b]) > 0.75]
    lines.append("**Redundant pairs:** " + (", ".join(f"{a}~{b} ({r})" for a, b, r in flagged)
                                            if flagged else "none at |rho|>0.75"))
    lines.append("")
    lines.append("## Standalone TB win rates (decided signals only; production barriers)")
    lines.append("")
    lines.append("| indicator | n_decided (sum) | mean win rate |")
    lines.append("|---|---|---|")
    for name, entries in sorted(all_win.items()):
        ns = sum(e["n_decided"] for e in entries)
        rates = [e["win_rate"] for e in entries if e["win_rate"] is not None]
        mean_rate = round(float(np.mean(rates)), 3) if rates else None
        lines.append(f"| {name} | {ns} | {mean_rate} |")
    lines.append("")
    lines.append("_Report-only (D3): breakeven at tp1.5/sl1.0 is 40% win rate. "
                 "Nothing in production reads this file._")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[indicator-corr] -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
