#!/usr/bin/env python
"""Offline evidence study for the v3.5 sentiment features (network, manual).

The backtest harness replays indicator+gate only — a brain voter can't be
A/B'd end-to-end offline. What CAN be measured honestly, and is here:

* daily market features (F&G level/roc/extreme, mempool fee-pressure z,
  tx momentum, price-vs-usage divergence) -> Spearman IC vs BTC forward
  1d / 7d returns over ~2 years;
* the classic contrarian table: forward returns after extreme-fear /
  extreme-greed days;
* per-pair taker-buy-ratio z (the live feature math) -> forward-return IC
  and quintile spread at 1h/4h/1d over the 12-pair evidence set.

Writes docs/sentiment-evidence.md. Usage:
    python scripts/analyze_sentiment.py [--pairs BTCUSDT,ETHUSDT,...]
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

FNG_URL = "https://api.alternative.me/fng/"
CHART_URL = "https://api.blockchain.info/charts/{name}"
KLINES_URL = "https://api.binance.com/api/v3/klines"
PAIRS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT",
         "DOGEUSDT", "AVAXUSDT", "LINKUSDT", "NEARUSDT", "ARBUSDT", "GALAUSDT"]
HORIZON = {"1h": 3, "4h": 2, "1d": 1}
OUT_MD = "docs/sentiment-evidence.md"

DAY_MS = 86_400_000


def _get(url, params=None):
    r = requests.get(url, params=params or {}, timeout=15,
                     headers={"User-Agent": "BitReinforceX/1.0"})
    r.raise_for_status()
    return r.json()


def _daily_map(values):
    """blockchain.info x/y series -> {utc_day: y} (last sample of the day)."""
    out = {}
    for p in values:
        out[int(p["x"]) // 86400] = float(p["y"])
    return out


def _series(day_map, days):
    return np.array([day_map.get(d, np.nan) for d in days], dtype=float)


def _z_roll(a, win=30):
    out = np.full(len(a), np.nan)
    for i in range(win, len(a)):
        w = a[i - win:i]
        m, s = np.nanmean(w), np.nanstd(w, ddof=1)
        if s > 1e-12 and not math.isnan(a[i]):
            out[i] = math.tanh((a[i] - m) / s / 2.0)
    return out


def _roc(a, back):
    out = np.full(len(a), np.nan)
    for i in range(back, len(a)):
        if not (math.isnan(a[i]) or math.isnan(a[i - back])) and abs(a[i - back]) > 1e-12:
            out[i] = (a[i] - a[i - back]) / abs(a[i - back])
    return out


def _ic(feat, fwd):
    m = ~(np.isnan(feat) | np.isnan(fwd))
    if m.sum() < 50:
        return None, 0
    rho, p = spearmanr(feat[m], fwd[m])
    return (rho, p), int(m.sum())


def _fmt_ic(res):
    if res[0] is None:
        return "n<50"
    (rho, p), n = res
    star = " *" if p < 0.05 else ""
    return f"{rho:+.3f} (p={p:.3f}, n={n}){star}"


def daily_study(md):
    print("[daily] fetching F&G + on-chain 2y ...")
    fng_rows = _get(FNG_URL, {"limit": 0, "format": "json"})["data"]
    fng = {int(r["timestamp"]) // 86400: float(r["value"]) for r in fng_rows}
    mem = _daily_map(_get(CHART_URL.format(name="mempool-size"),
                          {"timespan": "2years", "format": "json"})["values"])
    ntx = _daily_map(_get(CHART_URL.format(name="n-transactions"),
                          {"timespan": "2years", "format": "json"})["values"])
    txv = _daily_map(_get(CHART_URL.format(name="estimated-transaction-volume-usd"),
                          {"timespan": "2years", "format": "json"})["values"])

    kl = _get(KLINES_URL, {"symbol": "BTCUSDT", "interval": "1d", "limit": 1000})
    px = {int(k[0]) // DAY_MS: float(k[4]) for k in kl[:-1]}         # closed days

    days = sorted(set(px) & set(fng))[-730:]
    close = _series(px, days)
    f = _series(fng, days)

    feats = {
        "fng_level": (f - 50.0) / 50.0,
        "fng_roc_7d": np.clip((f - np.roll(f, 7)) / 25.0, -1, 1),
        "fng_extreme": np.where(f <= 20, (20 - f) / 20.0,
                                np.where(f >= 80, -(f - 80) / 20.0, 0.0)),
        "fee_pressure_z": _z_roll(_series(mem, days)),
        "tx_momentum": np.tanh(3.0 * _roc(_series(ntx, days), 7)),
        "onchain_divergence": np.tanh(3.0 * (_roc(_series(txv, days), 7)
                                             - _roc(close, 7))),
    }
    feats["fng_roc_7d"][:7] = np.nan

    md.append("## Daily market features vs BTC forward returns (2y)\n")
    md.append("| feature | IC vs fwd 1d | IC vs fwd 7d |")
    md.append("|---|---|---|")
    fwd1 = np.append(close[1:] / close[:-1] - 1.0, np.nan)
    fwd7 = np.append(close[7:] / close[:-7] - 1.0, [np.nan] * 7)
    for name, x in feats.items():
        md.append(f"| {name} | {_fmt_ic(_ic(x, fwd1))} | {_fmt_ic(_ic(x, fwd7))} |")

    md.append("\n### Contrarian table — forward BTC return after extreme days\n")
    md.append("| state | days | fwd 7d mean | fwd 7d >0 | fwd 30d mean |")
    md.append("|---|---|---|---|---|")
    fwd30 = np.append(close[30:] / close[:-30] - 1.0, [np.nan] * 30)
    for label, mask in (("extreme fear (F&G<20)", f < 20),
                        ("neutral (40–60)", (f >= 40) & (f <= 60)),
                        ("extreme greed (F&G>80)", f > 80)):
        m7, m30 = fwd7[mask], fwd30[mask]
        m7, m30 = m7[~np.isnan(m7)], m30[~np.isnan(m30)]
        if len(m7) == 0:
            md.append(f"| {label} | 0 | – | – | – |")
            continue
        md.append(f"| {label} | {len(m7)} | {m7.mean()*100:+.2f}% | "
                  f"{(m7 > 0).mean()*100:.0f}% | "
                  f"{(m30.mean()*100 if len(m30) else float('nan')):+.2f}% |")
    print("[daily] done")


def taker_study(md, pairs):
    md.append("\n## Taker buy-ratio z (live feature math) vs forward returns\n")
    md.append("| tf | pooled IC (fwd k bars) | Q5−Q1 spread | n |")
    md.append("|---|---|---|---|")
    for tf, k in HORIZON.items():
        feats_all, fwd_all = [], []
        for pair in pairs:
            try:
                rows = []
                end = None
                for _ in range(18 if tf == "1h" else 5):
                    params = {"symbol": pair, "interval": tf, "limit": 1000}
                    if end:
                        params["endTime"] = end
                    batch = _get(KLINES_URL, params)
                    if not batch:
                        break
                    rows = batch + rows
                    end = int(batch[0][0]) - 1
                    time.sleep(0.15)
                rows = rows[:-1]                       # drop open candle
                close = np.array([float(r[4]) for r in rows])
                vol = np.array([float(r[5]) for r in rows])
                tb = np.array([float(r[9]) for r in rows])
                ratio = np.where(vol > 0, tb / vol, 0.5)
                z = _z_roll(ratio)
                fwd = np.full(len(close), np.nan)
                fwd[:-k] = close[k:] / close[:-k] - 1.0
                feats_all.append(z)
                fwd_all.append(fwd)
                print(f"[taker] {pair} {tf}: {len(rows)} bars")
            except Exception as e:
                print(f"[taker] {pair} {tf} failed: {e}", file=sys.stderr)
        if not feats_all:
            continue
        x = np.concatenate(feats_all)
        y = np.concatenate(fwd_all)
        res = _ic(x, y)
        m = ~(np.isnan(x) | np.isnan(y))
        xq, yq = x[m], y[m]
        q = np.nanquantile(xq, [0.2, 0.8])
        spread = yq[xq >= q[1]].mean() - yq[xq <= q[0]].mean()
        md.append(f"| {tf} (k={k}) | {_fmt_ic(res)} | {spread*100:+.3f}% | {m.sum()} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default=",".join(PAIRS))
    args = ap.parse_args()
    pairs = [p.strip().upper() for p in args.pairs.split(",") if p.strip()]

    md = ["# Sentiment-feature evidence (offline IC study)\n",
          f"_Generated {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())} · "
          "Spearman IC, * = p<0.05. This measures FEATURES vs forward returns; "
          "the voter itself is judged live in shadow (harness replays "
          "indicator+gate only — stated per project honesty rule)._\n"]
    daily_study(md)
    taker_study(md, pairs)
    md.append("\n_Reading: |IC| 0.02–0.05 is normal for a single crypto feature; "
              "the bandit weighs and signs features from graded outcomes — this "
              "table only checks none of them is pure noise before go-live._\n")
    os.makedirs("docs", exist_ok=True)
    with open(OUT_MD, "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"wrote {OUT_MD}")
    return 0


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.exit(main())
