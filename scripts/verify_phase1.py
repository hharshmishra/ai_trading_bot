#!/usr/bin/env python3
"""Phase 1 headline metric: LLM calls per cycle, OLD per-coin vs NEW shared-context.

Runs the full 48-pair universe through the brain with a deterministic MOCK LLM
(no network, no API key, no spend) and prints the per-cycle call count both ways.
This is the concrete version of the plan's "~576 -> ~73 calls/cycle" claim.

    python scripts/verify_phase1.py
"""
from __future__ import annotations

import os
import sys
import tempfile
import types
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.modules.setdefault("pandas_ta", types.ModuleType("pandas_ta"))  # locally broken; unused here

# Reuse the exact test doubles so this script and the test agree.
sys.path.insert(0, str(ROOT / "tests"))
from test_phase1_shared_context import MockLLM, FakeIndicator, _fake_ohlcv  # noqa: E402

SYMBOLS = [s + "USDT" for s in [
    "AAVE", "ADA", "ALGO", "AR", "ARB", "ATOM", "AVAX", "AXS", "BCH", "BNB",
    "BTC", "CAKE", "COMP", "CRV", "DOGE", "DOT", "DYDX", "ENJ", "ETC", "ETH",
    "FET", "FIL", "FLOW", "GALA", "GMT", "GRT", "ICP", "IMX", "INJ", "LINK",
    "LRC", "LUNA", "MANA", "MKR", "NEAR", "OP", "POL", "PYTH", "RENDER", "SAND",
    "SHIB", "SNX", "SOL", "STORJ", "THETA", "UNI", "WLD", "XRP",
]]
TF = "4h"


def main() -> None:
    os.chdir(tempfile.mkdtemp())
    os.makedirs("logs", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    from agents import llm_client
    mock = MockLLM()
    llm_client.set_client(mock)

    from utils.data_fetcher import DataFetcher
    DataFetcher.get_ohlcv = lambda self, s, tf, limit=500: _fake_ohlcv(s, tf, limit)

    import agents.news_agent as na
    na.NewsRL.select_action = lambda self, feats: int(
        max(range(3), key=lambda i: self._logits(feats)[i]))

    from brain.decision_maker import DecisionMaker
    from market_context import build_market_context

    dm = DecisionMaker(prefer_csv=False)
    dm.indicator = FakeIndicator()
    dm.research._rl.policy.epsilon = 0.0
    dm.news._rl.policy.epsilon = 0.0

    # OLD: per-coin path.
    mock.reset_count()
    for sym in SYMBOLS:
        dm.decide(sym, TF)
    old_total = mock.call_count

    # NEW: shared context built once, then 1 call per coin.
    mock.reset_count()
    ctx = build_market_context(TF, SYMBOLS, dm.indicator, dm.news, dm.research)
    build_cost = mock.call_count
    per_coin = 0
    for sym in SYMBOLS:
        before = mock.call_count
        dm.decide(sym, TF, market_context=ctx)
        per_coin += mock.call_count - before
    new_total = build_cost + per_coin

    print("\n  Phase 1 — LLM calls per cycle ({} pairs, 1 timeframe)".format(len(SYMBOLS)))
    print("  " + "-" * 52)
    print(f"  OLD  (per-coin)        : {old_total:>5}  ({old_total/len(SYMBOLS):.1f}/pair)")
    print(f"  NEW  (shared context)  : {new_total:>5}  "
          f"(build {build_cost} + {per_coin/len(SYMBOLS):.0f}/pair)")
    print(f"  reduction              : {old_total/max(new_total,1):.1f}x  "
          f"({100*(1-new_total/old_total):.0f}% fewer)")
    print()


if __name__ == "__main__":
    main()
