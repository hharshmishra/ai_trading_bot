"""v3.5 SentimentAgent: feature truth-table, bandit persistence, decide()
contract, fetcher parsing/caching on canned fixtures, Telegram note. No
network — every fetch is monkeypatched or served from fixtures."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from agents import sentiment_agent as sa
from agents.sentiment_agent import (N_FEATURES, SentimentAgent, SentimentRL,
                                    build_features, sentiment_note)
from utils import sentiment_fetcher as sfx


def _bundle(**kw):
    b = {"fng": 50.0, "fng_hist": [50.0] * 45,
         "mempool": [100.0] * 30, "ntx": [1000.0] * 30,
         "txvol": [5e9] * 30, "trending": {"PEPE", "SOL"}}
    b.update(kw)
    return b


def _flow(ratio=0.5, n=40, last_ratio=None):
    """n closed candles with constant taker-buy ratio; optionally spike the last."""
    rows = []
    for i in range(n):
        r = last_ratio if (last_ratio is not None and i == n - 1) else ratio
        rows.append((1e12 + i * 3.6e6, 100.0, 100.0 * r))
    return rows


class TestFeatures:
    def test_bounds_and_length(self):
        f = build_features(_bundle(), _flow(), 0.05, "BTC")
        assert len(f) == N_FEATURES
        assert all(-1.0 <= x <= 1.0 for x in f)

    def test_missing_everything_degrades_to_zero(self):
        f = build_features(None, None, None, "BTC")
        assert f[:9] == [0.0] * 9 and f[9] == 0.2      # only bias survives

    def test_fng_level_and_extremes(self):
        assert build_features(_bundle(fng=10.0), None, None)[0] == pytest.approx(-0.8)
        assert build_features(_bundle(fng=10.0), None, None)[2] == pytest.approx(0.5)
        assert build_features(_bundle(fng=90.0), None, None)[2] == pytest.approx(-0.5)
        assert build_features(_bundle(fng=50.0), None, None)[2] == 0.0

    def test_fng_roc_sign(self):
        rising = _bundle(fng_hist=[20.0] * 38 + [20, 25, 30, 35, 40, 45, 50])
        assert build_features(rising, None, None)[1] > 0

    def test_fee_pressure_z_sign(self):
        noisy = [100.0 + (3.0 if i % 2 else -3.0) for i in range(29)]
        hot = _bundle(mempool=noisy + [400.0])
        quiet = _bundle(mempool=noisy + [10.0])
        assert build_features(hot, None, None)[3] > 0.3
        assert build_features(quiet, None, None)[3] < -0.3

    def test_hollow_rally_divergence_negative(self):
        # price +20% while on-chain volume flat -> negative divergence
        f = build_features(_bundle(), None, 0.20)
        assert f[5] < -0.3

    def test_silent_accumulation_divergence_positive(self):
        surge = _bundle(txvol=[5e9] * 25 + [8e9] * 5)   # usage up, price flat
        assert build_features(surge, None, 0.0)[5] > 0.3

    def test_trending_hit_matches_base_ticker(self):
        assert build_features(_bundle(), None, None, "PEPE")[6] == 1.0
        assert build_features(_bundle(), None, None, "BTC")[6] == 0.0

    def test_taker_flow_spike_and_trend(self):
        base = [(1e12 + i * 3.6e6, 100.0,
                 100.0 * (0.52 if i % 2 else 0.48)) for i in range(39)]
        spike = build_features(None, base + [(1e12, 100.0, 75.0)], None)
        assert spike[7] > 0.3                           # buy-side z positive
        shifted = build_features(None, _flow(0.4)[:35] + _flow(0.65)[:5], None)
        assert shifted[8] > 0.3                         # recent flow > prior


class TestRL:
    def test_update_persists_and_reloads(self, tmp_path):
        p = str(tmp_path / "pol.json")
        rl = SentimentRL(policy_path=p)
        before = [row[:] for row in rl.weights]
        rl.update([0.5] * N_FEATURES, 2, 1.0)
        assert rl.weights != before
        assert SentimentRL(policy_path=p).weights == rl.weights

    def test_reward_raises_action_prob(self, tmp_path):
        rl = SentimentRL(policy_path=str(tmp_path / "pol.json"))
        feats = [0.5] * N_FEATURES
        p0 = rl.prob(feats, 2)
        for _ in range(5):
            rl.update(feats, 2, 1.0)
        assert rl.prob(feats, 2) > p0


class TestAgent:
    @pytest.fixture
    def agent(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sa, "POLICY_PATH", str(tmp_path / "pol.json"))
        monkeypatch.setattr(sfx, "fetch_market_sentiment", lambda: _bundle())
        monkeypatch.setattr(sfx, "fetch_taker_flow", lambda s, tf: _flow(0.6))
        ag = SentimentAgent(data_fetcher=object())      # fetcher never touched:
        monkeypatch.setattr(ag, "_btc_roc_7d", lambda: 0.02)
        ag._rl = SentimentRL(policy_path=str(tmp_path / "pol.json"))
        ag._rl.epsilon = 0.0
        return ag

    def test_available_path_contract(self, agent):
        out = agent.decide("BTCUSDT", "1h")
        assert out["available"] is True
        assert out["action"] in ("buy", "sell", "skip")
        assert 0.5 <= out["confidence"] <= 0.9
        assert len(out["rl"]["feats"]) == N_FEATURES
        assert out["details"]["taker_buy_pct"] == pytest.approx(0.6)

    def test_total_outage_is_exact_noop(self, agent, monkeypatch):
        monkeypatch.setattr(sfx, "fetch_market_sentiment", lambda: None)
        monkeypatch.setattr(sfx, "fetch_taker_flow", lambda s, tf: None)
        out = agent.decide("BTCUSDT", "4h")
        assert out == {"agent": "sentiment_agent", "chartName": "BTCUSDT",
                       "timeframe": "4h", "action": "skip", "confidence": 0.0,
                       "available": False, "rl": None}

    def test_partial_outage_still_votes(self, agent, monkeypatch):
        monkeypatch.setattr(sfx, "fetch_market_sentiment", lambda: None)
        out = agent.decide("ETHUSDT", "1h")               # taker flow alone
        assert out["available"] is True

    def test_apply_reward_none_is_noop(self, agent):
        before = [row[:] for row in agent._rl.weights]
        agent.apply_reward(None, None, 1.0)
        assert agent._rl.weights == before


class TestFetcherParsing:
    def test_fng_parses_string_values_and_reverses(self, monkeypatch):
        js = {"data": [{"value": "23", "timestamp": "1783209600"},
                       {"value": "30", "timestamp": "1783123200"}]}   # newest first
        monkeypatch.setattr(sfx, "_get_json", lambda url, params=None: js)
        cur, hist = sfx._fng()
        assert cur == 23.0 and hist == [30.0, 23.0]       # oldest -> newest

    def test_chart_parses_xy_series(self, monkeypatch):
        js = {"values": [{"x": 1, "y": 10.0}, {"x": 2, "y": 12.5}]}
        monkeypatch.setattr(sfx, "_get_json", lambda url, params=None: js)
        assert sfx._chart("mempool-size") == [10.0, 12.5]

    def test_taker_flow_drops_open_candle(self, monkeypatch):
        import time as _t
        now_ms = _t.time() * 1000
        closed = [now_ms - 7.2e6, "1", "1", "1", "1", "100", now_ms - 3.6e6,
                  "0", 0, "60", "0", "0"]
        open_k = [now_ms - 3.6e6, "1", "1", "1", "1", "50", now_ms + 3.6e6,
                  "0", 0, "10", "0", "0"]
        monkeypatch.setattr(sfx, "_get_json", lambda url, params=None: [closed, open_k])
        sfx.clear_cache()
        rows = sfx.fetch_taker_flow("BTCUSDT", "1h")
        assert len(rows) == 1 and rows[0][1] == 100.0 and rows[0][2] == 60.0

    def test_partial_bundle_survives_one_dead_source(self, monkeypatch):
        def _get(url, params=None):
            if "alternative.me" in url:
                raise RuntimeError("down")
            if "blockchain.info" in url:
                return {"values": [{"x": i, "y": float(i)} for i in range(30)]}
            return {"coins": [{"item": {"symbol": "sol"}}]}
        monkeypatch.setattr(sfx, "_get_json", _get)
        sfx.clear_cache()
        b = sfx.fetch_market_sentiment()
        assert b["fng"] is None and b["mempool"] is not None
        assert b["trending"] == {"SOL"}

    def test_bundle_cache_ttl(self, monkeypatch):
        calls = {"n": 0}

        def _get(url, params=None):
            calls["n"] += 1
            return {"data": [{"value": "50"}], "values": [], "coins": []}
        monkeypatch.setattr(sfx, "_get_json", _get)
        sfx.clear_cache()
        sfx.fetch_market_sentiment()
        first = calls["n"]
        sfx.fetch_market_sentiment()                      # served from cache
        assert calls["n"] == first


class TestNote:
    def test_note_formats(self):
        n = sentiment_note({"fng": 12.0, "taker_buy_pct": 0.63,
                            "fee_pressure_z": 0.8, "trending_hit": True})
        assert "F&G 12 (extreme fear)" in n and "taker 63% buy" in n
        assert "fees hot" in n and "trending" in n

    def test_note_none_on_empty(self):
        assert sentiment_note(None) is None
        assert sentiment_note({}) is None
