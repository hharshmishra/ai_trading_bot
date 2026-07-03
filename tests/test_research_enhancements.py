"""Phase B: macro price scoring, money-flow v2, ecosystem cache. Offline."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config


class TestTrendScore:
    def test_scoring_math(self):
        from utils.macro_prices import _trend_score
        s = pd.Series(range(100, 131), dtype=float)   # steady climb
        # last=130, 5 sessions back=125 -> +4% -> /0.03 clipped to 1
        assert _trend_score(s, days=5, scale=0.03) == 1.0
        flat = pd.Series([100.0] * 10)
        assert _trend_score(flat, days=5, scale=0.03) == 0.0
        assert _trend_score(None) is None
        assert _trend_score(pd.Series([1.0, 2.0])) is None   # too short

    def test_dxy_requires_fred_key(self, monkeypatch):
        from utils import macro_prices
        monkeypatch.delenv("FRED_API_KEY", raising=False)
        assert macro_prices.dxy_score() is None   # no key -> None, no network


class TestLogic2And5PriceBlend:
    def _agent(self):
        from agents.research_agent import ResearchAgent
        return ResearchAgent()

    class _News:
        def run(self, pair):
            return {"pair_json": {"sentiment": "bullish", "confidence": 0.5},
                    "overall_json": {}}

    def test_spx_blend(self):
        a = self._agent()
        score, d = a._logic2_spx(self._News(), price_score=1.0)
        assert score == pytest.approx(0.6 * 1.0 + 0.4 * 0.5)
        assert d["source"] == "price+news"

    def test_spx_news_only_unchanged(self):
        a = self._agent()
        score, _ = a._logic2_spx(self._News(), price_score=None)
        assert score == pytest.approx(0.5)   # old behavior exactly

    def test_dxy_negation(self):
        a = self._agent()
        # strong dollar (price +1) + bullish-dollar news (+0.5) -> negative for crypto
        score, d = a._logic5_dxy(self._News(), price_score=1.0)
        assert score == pytest.approx(-(0.6 + 0.2))
        score2, _ = a._logic5_dxy(self._News(), price_score=None)
        assert score2 == pytest.approx(-0.5)

    def test_price_only_when_news_fails(self):
        a = self._agent()
        score, d = a._logic2_spx(None, price_score=0.5)
        assert score == pytest.approx(0.3) and d["source"] == "price_only"


class TestMoneyFlowV2:
    def _agent(self):
        from agents.research_agent import ResearchAgent
        return ResearchAgent()

    class _Ind:
        """Deterministic indicator: BTC up, ETHBTC flat, alts down."""
        def __init__(self, table):
            self.table = table
        def decide(self, pair, tf):
            act, conf = self.table.get(pair, ("skip", 0.5))
            return {"action": act, "confidence": conf}

    def test_risk_off(self):
        ind = self._Ind({"BTCUSDT": ("sell", 0.8)})
        score, d = self._agent()._logic3_money_flow_v2("4h", ind, dom_level=58.0, dom_roc=0.02)
        assert d["phase"] == "risk_off" and score == -1.0

    def test_btc_led(self):
        ind = self._Ind({"BTCUSDT": ("buy", 0.8)})
        score, d = self._agent()._logic3_money_flow_v2("4h", ind, dom_level=58.0, dom_roc=0.02)
        assert d["phase"] == "btc_led" and score == -0.8

    def test_alt_breadth(self):
        ind = self._Ind({"BTCUSDT": ("skip", 0.5),
                         "LINKUSDT": ("buy", 0.8), "POLUSDT": ("buy", 0.8),
                         "SOLUSDT": ("buy", 0.8), "BNBUSDT": ("buy", 0.8),
                         "XRPUSDT": ("buy", 0.8)})
        score, d = self._agent()._logic3_money_flow_v2("4h", ind, dom_level=48.0, dom_roc=-0.02)
        assert d["phase"] == "alt_breadth" and score == 0.7

    def test_blend_fallback(self):
        ind = self._Ind({})
        score, d = self._agent()._logic3_money_flow_v2("4h", ind, dom_level=None, dom_roc=None)
        assert d["phase"] == "blend" and -1 <= score <= 1


class TestEcosystemCache:
    def test_fresh_cache_overlays_membership(self, tmp_path, monkeypatch):
        import agents.research_agent as ra
        monkeypatch.setattr(config, "ECOSYSTEMS_AUTO", True)
        original = list(ra.ECOSYSTEMS["gaming"])
        cache = tmp_path / "eco.json"
        cache.write_text(json.dumps({"fetched_ts": time.time(),
                                     "ecosystems": {"gaming": ["gala", "axs", "sand"]}}))
        try:
            assert ra.load_ecosystems_cache(str(cache)) is True
            assert ra.ECOSYSTEMS["gaming"] == ["GALA", "AXS", "SAND"]
        finally:
            ra.ECOSYSTEMS["gaming"] = original

    def test_stale_cache_ignored(self, tmp_path, monkeypatch):
        import agents.research_agent as ra
        monkeypatch.setattr(config, "ECOSYSTEMS_AUTO", True)
        cache = tmp_path / "eco.json"
        cache.write_text(json.dumps({"fetched_ts": time.time() - 8 * 86400,
                                     "ecosystems": {"gaming": ["XXX"]}}))
        assert ra.load_ecosystems_cache(str(cache)) is False
        assert "XXX" not in ra.ECOSYSTEMS["gaming"]

    def test_flag_off_noop(self, tmp_path, monkeypatch):
        import agents.research_agent as ra
        monkeypatch.setattr(config, "ECOSYSTEMS_AUTO", False)
        cache = tmp_path / "eco.json"
        cache.write_text(json.dumps({"fetched_ts": time.time(),
                                     "ecosystems": {"gaming": ["XXX"]}}))
        assert ra.load_ecosystems_cache(str(cache)) is False
