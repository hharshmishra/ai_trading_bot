"""Phase 1 verification: shared market-context cuts LLM cost ~8x with IDENTICAL
behaviour.

What this proves
----------------
1. COST: with a shared MarketContext, each pair's brain.decide costs exactly ONE
   LLM call (its own pair news scan); without it, each pair costs >=8 (the old
   per-coin explosion: driver/SPX/DXY news x2 each + the brain's own news x2).
2. EQUIVALENCE: research's 10-feature vector and the brain's final action are
   bit-for-bit identical between the old per-coin path and the new shared-context
   path — the shared context only RELOCATES the same computations.

Test doubles (Phase 1 touches none of these):
  * MockLLM   — deterministic JSON per prompt; counts calls. No network/key.
  * FakeIndicator — deterministic action/confidence; avoids the (locally broken)
                    pandas_ta dependency. Phase 1 does not modify the indicator.
  * Monkeypatched DataFetcher.get_ohlcv — canned OHLCV; no network.
  * NewsRL.select_action forced to argmax + research epsilon=0 — determinism.
"""
from __future__ import annotations

import hashlib
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make project root importable when run from anywhere.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# indicator_agent imports pandas_ta at module top; it's broken in this env and we
# replace the indicator with a deterministic fake anyway, so stub the module so
# the brain import succeeds. ta is only referenced inside decide(), never called.
try:  # pragma: no cover - environment dependent
    import pandas_ta  # noqa: F401
except Exception:  # pragma: no cover
    sys.modules["pandas_ta"] = types.ModuleType("pandas_ta")


SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT", "LINKUSDT", "ADAUSDT"]
TF = "4h"


# --------------------------------------------------------------------------- #
# Deterministic test doubles
# --------------------------------------------------------------------------- #
def _stable_hash(s: str) -> int:
    return int(hashlib.sha256(s.encode()).hexdigest(), 16)


class MockLLM:
    """Counts calls; returns deterministic JSON keyed by prompt content."""

    def __init__(self):
        self._n = 0

    def chat_json(self, prompt: str) -> dict:
        self._n += 1
        if "panic-worthy" in prompt:  # the OVERALL scan prompt
            return {"has_panic": False, "sentiment": "Neutral",
                    "confidence": 0.5, "top_headlines": []}
        # PAIR scan prompt — deterministic sentiment from the (pair-embedded) text
        h = _stable_hash(prompt)
        sentiment = ["Bullish", "Bearish", "Neutral"][h % 3]
        confidence = round(0.5 + (h % 50) / 100.0, 2)
        return {"pair": "NA", "sentiment": sentiment,
                "confidence": confidence, "top_headlines": []}

    @property
    def call_count(self) -> int:
        return self._n

    def reset_count(self) -> None:
        self._n = 0


@dataclass
class FakeDecision:
    agent: str = "indicator_agent"
    chartName: str = ""
    timeframe: str = ""
    action: str = "skip"
    confidence: float = 0.6
    details: dict = field(default_factory=dict)


class FakeIndicator:
    """Deterministic indicator agent (no pandas_ta). Same output for same pair."""

    def decide(self, symbol: str, timeframe: str, ohlcv=None, limit: int = 500):
        h = _stable_hash(f"{symbol}|{timeframe}")
        action = ["buy", "sell", "skip"][h % 3]
        conf = round(0.55 + (h % 40) / 100.0, 4)
        return FakeDecision(chartName=symbol, timeframe=timeframe, action=action,
                            confidence=conf,
                            details={"direct_signals": [
                                {"name": "nwe", "signal": action, "confidence": conf}]})

    def learn(self, *a, **k):  # never invoked in this test
        pass


def _fake_ohlcv(symbol: str, timeframe: str, limit: int = 500) -> pd.DataFrame:
    n = 300
    seed = _stable_hash(f"{symbol}|{timeframe}") % (2 ** 32)
    rng = np.random.default_rng(seed)
    close = np.abs(100 + (seed % 100) + rng.normal(0, 1, n).cumsum()) + 1.0
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="h"),
        "open": close, "high": close * 1.01, "low": close * 0.99,
        "close": close, "volume": rng.uniform(100, 1000, n),
    })


# --------------------------------------------------------------------------- #
# Fixture: isolated cwd, mock LLM, deterministic RL, fake data + indicator
# --------------------------------------------------------------------------- #
@pytest.fixture()
def dm_and_mock(tmp_path, monkeypatch):
    # Isolate logs/data writes into a temp cwd (keep the real project logs clean).
    monkeypatch.chdir(tmp_path)
    (tmp_path / "logs").mkdir()
    (tmp_path / "data").mkdir()

    # Route all LLM traffic to the deterministic mock.
    from agents import llm_client
    mock = MockLLM()
    llm_client.set_client(mock)

    # Canned OHLCV for every symbol (no network).
    from utils.data_fetcher import DataFetcher
    monkeypatch.setattr(DataFetcher, "get_ohlcv",
                        lambda self, s, tf, limit=500: _fake_ohlcv(s, tf, limit))

    # Force the news action selection to deterministic argmax (it otherwise
    # samples from the softmax even at epsilon=0).
    import agents.news_agent as na
    monkeypatch.setattr(na.NewsRL, "select_action",
                        lambda self, feats: int(max(range(3), key=lambda i: self._logits(feats)[i])))

    from brain.decision_maker import DecisionMaker
    dm = DecisionMaker(prefer_csv=False)
    dm.indicator = FakeIndicator()             # deterministic, no pandas_ta
    dm.research._rl.policy.epsilon = 0.0        # argmax => deterministic action
    dm.news._rl.policy.epsilon = 0.0
    return dm, mock


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def _research_feats(decision: dict) -> list:
    return list(decision["agents"]["research"]["raw"]["details"]["features"].values())


def test_shared_context_is_cheaper_and_equivalent(dm_and_mock):
    from market_context import build_market_context

    dm, mock = dm_and_mock

    # ---- OLD path: per-coin (no shared context) -------------------------- #
    old_actions, old_feats, old_marginals = {}, {}, {}
    for sym in SYMBOLS:
        mock.reset_count()
        res = dm.decide(sym, TF)                 # market_context=None
        old_marginals[sym] = mock.call_count
        old_actions[sym] = res["final"]["action"]
        old_feats[sym] = _research_feats(res)

    # ---- NEW path: shared context built once, then reused ---------------- #
    mock.reset_count()
    ctx = build_market_context(TF, SYMBOLS, dm.indicator, dm.news, dm.research)
    build_cost = mock.call_count

    new_actions, new_feats, new_marginals = {}, {}, {}
    for sym in SYMBOLS:
        mock.reset_count()
        res = dm.decide(sym, TF, market_context=ctx)
        new_marginals[sym] = mock.call_count
        new_actions[sym] = res["final"]["action"]
        new_feats[sym] = _research_feats(res)

    # 1) COST: every old pair cost >=8 LLM calls; every new pair costs exactly 1.
    for sym in SYMBOLS:
        assert old_marginals[sym] >= 8, f"{sym}: old marginal {old_marginals[sym]} (<8)"
        assert new_marginals[sym] == 1, f"{sym}: new marginal {new_marginals[sym]} (!=1)"

    old_total = sum(old_marginals.values())
    new_total = build_cost + sum(new_marginals.values())
    assert new_total < old_total
    # Shared cost amortises: the new whole-cycle total is far below the old one.
    assert new_total <= old_total * 0.6, (old_total, new_total, build_cost)

    # 2) EQUIVALENCE: research features identical (bit-for-bit) per symbol.
    for sym in SYMBOLS:
        assert old_feats[sym] == pytest.approx(new_feats[sym], abs=1e-9), sym

    # 3) EQUIVALENCE: brain's final action identical per symbol.
    for sym in SYMBOLS:
        assert old_actions[sym] == new_actions[sym], (
            sym, old_actions[sym], new_actions[sym])


def test_news_overall_scan_is_reused(dm_and_mock):
    """A per-symbol news run with an injected overall scan costs 1 call, not 2."""
    dm, mock = dm_and_mock
    overall = dm.news.scan_overall().model_dump()

    mock.reset_count()
    dm.news.run("BTCUSDT", overall_json=overall)
    assert mock.call_count == 1                  # pair scan only

    mock.reset_count()
    dm.news.run("BTCUSDT")                        # no injected overall
    assert mock.call_count == 2                  # overall + pair
