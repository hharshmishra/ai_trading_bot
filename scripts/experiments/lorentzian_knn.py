#!/usr/bin/env python
"""Lorentzian kNN classifier — EXPERIMENT ONLY (enhancement D5, TEST tier).

Never wired into production. Reproduces the jdehorty-style approach: kNN over
[RSI, ADX, CCI, WT-proxy] feature space using LORENTZIAN distance
(sum(log(1+|dx|)) — robust to outlier bars), labels = k-bar-forward direction,
walk-forward: train on the first 70% of each pair's history, evaluate on the
last 30%.

    python scripts/experiments/lorentzian_knn.py --pairs BTCUSDT,ETHUSDT --tf 4h

Verdict criteria (from the plan): adopt as a regime/entry filter only if
holdout precision beats the 40% TB breakeven with n >= 100 — else record the
numbers in the report and drop it.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd


def features(df: pd.DataFrame) -> pd.DataFrame:
    import pandas_ta as ta
    f = pd.DataFrame(index=df.index)
    f["rsi"] = ta.rsi(df["close"], length=14)
    adx = ta.adx(df["high"], df["low"], df["close"], length=14)
    f["adx"] = adx["ADX_14"]
    f["cci"] = ta.cci(df["high"], df["low"], df["close"], length=20)
    hlc3 = (df["high"] + df["low"] + df["close"]) / 3
    f["wt"] = (hlc3 - hlc3.rolling(10).mean()) / (0.015 * hlc3.rolling(10).std() + 1e-9)
    return f


def lorentzian_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.log1p(np.abs(a - b)).sum(axis=1)


def run_pair(df: pd.DataFrame, horizon: int = 4, k: int = 8,
             thresh: float = 0.004) -> dict:
    f = features(df)
    fwd = df["close"].shift(-horizon) / df["close"] - 1.0
    label = np.where(fwd > thresh, 1, np.where(fwd < -thresh, -1, 0))

    valid = f.notna().all(axis=1) & pd.notna(fwd)
    X = ((f - f.mean()) / (f.std() + 1e-9))[valid].to_numpy()
    y = label[valid.to_numpy()]
    cut = int(len(X) * 0.7)
    Xtr, ytr, Xte, yte = X[:cut], y[:cut], X[cut:], y[cut:]

    hits = preds = 0
    for i in range(len(Xte)):
        d = lorentzian_dist(Xtr, Xte[i])
        vote = ytr[np.argsort(d)[:k]].sum()
        if abs(vote) >= k * 0.5:                     # confident neighborhood only
            pred = 1 if vote > 0 else -1
            preds += 1
            hits += int(pred == yte[i])
    return {"n_train": cut, "n_pred": preds,
            "precision": round(hits / preds, 3) if preds else None}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", default="BTCUSDT,ETHUSDT")
    ap.add_argument("--tf", default="4h")
    ap.add_argument("--horizon", type=int, default=4)
    args = ap.parse_args()

    for pair in [p.strip().upper() for p in args.pairs.split(",")]:
        path = f"data/history/{pair}_{args.tf}.csv"
        if not os.path.exists(path):
            print(f"{pair}: no cached history — run a backtest first")
            continue
        df = pd.read_csv(path)
        out = run_pair(df, horizon=args.horizon)
        print(f"{pair} {args.tf}: {out}")
    print("verdict rule: adopt only if precision > 0.40 with n_pred >= 100 "
          "on BOTH majors — else drop (documented in the deck).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
