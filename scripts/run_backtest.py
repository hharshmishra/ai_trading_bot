#!/usr/bin/env python
"""Backtest CLI — replay the production indicator+gate pipeline over history.

Examples:
    python scripts/run_backtest.py --pairs BTCUSDT,ETHUSDT,SOLUSDT --tfs 1h,4h --start 2024-07-01
    python scripts/run_backtest.py --pairs all --tfs 1h,4h,1d --start 2024-07-01 --label baseline --workers 6
    python scripts/run_backtest.py ... --gate v2 --baseline logs/backtest/baseline/report.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.data import load_or_fetch  # noqa: E402
from backtest.metrics import compare, summarize  # noqa: E402
from backtest.report import write_report  # noqa: E402
from universe import SYMBOLS  # noqa: E402  — single source of truth (48 pairs + env add/remove)
from grader import HORIZON_K, THRESHOLD  # noqa: E402


def _replay_task(args_tuple):
    """Worker: (pair, tf, start, end, window, gate). Builds its own agent."""
    pair, tf, start, end, window, gate = args_tuple
    from agents.indicator_agent import IndicatorAgent
    from backtest.engine import replay_pair
    if gate == "v2":
        from signals import should_emit_signal_v2 as gate_fn  # Phase 2+
    else:
        from signals import should_emit_signal as gate_fn

    df = load_or_fetch(pair, tf, start, end)
    agent = IndicatorAgent()
    r = replay_pair(df, pair, tf, agent=agent, gate_fn=gate_fn,
                    k=HORIZON_K.get(tf, 1), theta=THRESHOLD.get(tf, 0.01),
                    window=window)
    return pair, tf, r.emissions, r.funnel, len(r.bars)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", default="BTCUSDT,ETHUSDT,SOLUSDT",
                    help="comma list, or 'all' for the 48-pair universe")
    ap.add_argument("--tfs", default="1h", help="comma list of timeframes")
    ap.add_argument("--start", required=True, help="ISO date, e.g. 2024-07-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--window", type=int, default=500)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--gate", choices=("v1", "v2"), default="v1")
    ap.add_argument("--label", default=None, help="run id; default = timestamp")
    ap.add_argument("--baseline", default=None,
                    help="path to a baseline report.json to compare against")
    ap.add_argument("--out-root", default="logs/backtest")
    args = ap.parse_args()

    pairs = SYMBOLS if args.pairs.strip().lower() == "all" else [
        p.strip().upper() for p in args.pairs.split(",") if p.strip()]
    tfs = [t.strip() for t in args.tfs.split(",") if t.strip()]
    label = args.label or f"run-{int(time.time())}"
    run_dir = os.path.join(args.out_root, label)

    # Download serially first (shared ccxt client + rate limits), replay in parallel.
    print(f"[backtest] caching history for {len(pairs)} pairs × {tfs} …")
    for pair in pairs:
        for tf in tfs:
            try:
                df = load_or_fetch(pair, tf, args.start, args.end)
                print(f"  {pair} {tf}: {len(df)} candles")
            except Exception as e:
                print(f"  {pair} {tf}: FAILED ({e})")

    tasks = [(pair, tf, args.start, args.end, args.window, args.gate)
             for pair in pairs for tf in tfs]
    emissions, funnel, bars_total, failures = [], {}, 0, []

    t0 = time.time()
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as exe:
            futs = {exe.submit(_replay_task, t): t for t in tasks}
            for fut in as_completed(futs):
                pair, tf, *_ = futs[fut]
                try:
                    _, _, ems, fun, nbars = fut.result()
                except Exception as e:
                    failures.append(f"{pair} {tf}: {e}")
                    continue
                emissions.extend(ems)
                bars_total += nbars
                for k_, v in fun.items():
                    funnel[k_] = funnel.get(k_, 0) + v
                print(f"  replayed {pair} {tf}: {nbars} bars, {len(ems)} emissions")
    else:
        for t in tasks:
            pair, tf = t[0], t[1]
            try:
                _, _, ems, fun, nbars = _replay_task(t)
            except Exception as e:
                failures.append(f"{pair} {tf}: {e}")
                continue
            emissions.extend(ems)
            bars_total += nbars
            for k_, v in fun.items():
                funnel[k_] = funnel.get(k_, 0) + v
            print(f"  replayed {pair} {tf}: {nbars} bars, {len(ems)} emissions")

    summary = summarize(emissions)
    comparison = None
    if args.baseline:
        with open(args.baseline) as f:
            comparison = compare(json.load(f)["summary"], summary)

    meta = {"label": label, "pairs": pairs, "tfs": tfs, "start": args.start,
            "end": args.end, "gate": args.gate, "window": args.window,
            "bars": bars_total, "failures": failures,
            "elapsed_s": round(time.time() - t0, 1)}
    md = write_report(run_dir, meta, summary, funnel, comparison)
    with open(os.path.join(run_dir, "emissions.json"), "w") as f:
        json.dump(emissions, f, default=str)

    print(f"[backtest] {len(emissions)} emissions from {bars_total} bars "
          f"in {meta['elapsed_s']}s → {md}")
    if failures:
        print(f"[backtest] {len(failures)} pair/tf failures: {failures[:5]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
