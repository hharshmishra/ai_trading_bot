"""Backtest harness tests (accuracy upgrade Phase 1). No network: synthetic OHLCV."""
import numpy as np
import pandas as pd
import pytest

from agents import custom_indicators as ci


def _synthetic_ohlcv(n=200, seed=42, freq="h"):
    rng = np.random.default_rng(seed)
    steps = rng.normal(0, 0.8, n)
    # inject a trend leg + a ranging leg so signals actually fire
    steps[n // 3: n // 2] += 0.9
    close = 100 + np.cumsum(steps)
    close = np.maximum(close, 5.0)
    high = close + np.abs(rng.normal(0.5, 0.3, n))
    low = close - np.abs(rng.normal(0.5, 0.3, n))
    opn = np.concatenate([[close[0]], close[:-1]])
    vol = np.abs(rng.normal(1000, 250, n)) + 1
    return pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=n, freq=freq),
        "open": opn, "high": high, "low": low, "close": close, "volume": vol,
    })


class TestNweVectorizationParity:
    def test_vectorized_matches_reference_loop(self):
        df = _synthetic_ohlcv(n=160)
        fast = ci.apply_nadaraya_watson_envelope(df.copy(), h=8.0, mult=3.0)
        ref = ci._nwe_repaint_reference(df.copy(), h=8.0, mult=3.0)
        for col in ("nwe_out", "nwe_mae", "nwe_upper", "nwe_lower"):
            assert np.allclose(fast[col].to_numpy(), ref[col].to_numpy(),
                               rtol=1e-10, atol=1e-8), col

    def test_vectorized_matches_reference_other_bandwidth(self):
        df = _synthetic_ohlcv(n=80, seed=7)
        fast = ci.apply_nadaraya_watson_envelope(df.copy(), h=4.0, mult=2.0)
        ref = ci._nwe_repaint_reference(df.copy(), h=4.0, mult=2.0)
        assert np.allclose(fast["nwe_out"].to_numpy(), ref["nwe_out"].to_numpy())
        assert np.allclose(fast["nwe_upper"].to_numpy(), ref["nwe_upper"].to_numpy())

    def test_non_repaint_branch_untouched(self):
        df = _synthetic_ohlcv(n=120)
        out = ci.apply_nadaraya_watson_envelope(df.copy(), repaint=False)
        assert "nwe_out" in out.columns  # endpoint branch still works


class TestEngineParity:
    """The engine's per-bar decision must equal calling the production agent
    directly on the same trailing window (parity by construction — this test
    guards regressions)."""

    def test_last_emission_matches_direct_decide(self, monkeypatch, tmp_path):
        from agents.indicator_agent import IndicatorAgent
        from backtest.engine import replay_pair, res_from_indicator
        from signals import should_emit_signal

        df = _synthetic_ohlcv(n=180, seed=11)
        window = 120
        agent = IndicatorAgent()

        result = replay_pair(df, "TESTUSDT", "1h", agent=agent, window=window,
                             warmup=window, k=3)
        # replay covers bars [warmup-1 .. n-1]; recompute one covered bar directly
        t = 150
        wdf = df.iloc[t - window + 1: t + 1].reset_index(drop=True)
        dec = agent.decide("TESTUSDT", "1h", ohlcv=wdf.copy(), log=False)
        res = res_from_indicator(dec)
        emit, overall, nwe, conf, reason = should_emit_signal(res)

        bar = [b for b in result.bars if b["t"] == t]
        assert len(bar) == 1
        assert bar[0]["emit"] == emit
        if emit:
            assert bar[0]["action"] == overall
            assert bar[0]["reason"] == reason

    def test_replay_produces_labeled_emissions(self):
        from agents.indicator_agent import IndicatorAgent
        from backtest.engine import replay_pair

        df = _synthetic_ohlcv(n=220, seed=3)
        agent = IndicatorAgent()
        result = replay_pair(df, "TESTUSDT", "1h", agent=agent, window=120,
                             warmup=120, k=3)
        assert result.bars, "replay should cover bars"
        for e in result.emissions:
            assert e["action"] in ("buy", "sell")
            assert e["label_tb"] in ("tp", "sl", "timeout", "incomplete")
            assert e["label_fixed"] in ("buy", "sell", "skip", None)
            assert e["entry"] > 0
            # tail emissions may lack full path; others carry barrier prices
            if e["label_tb"] in ("tp", "sl"):
                assert e["hit_idx"] >= 1

    def test_decide_log_kwarg_suppresses_jsonl(self, tmp_path, monkeypatch):
        import agents.indicator_agent as ia
        monkeypatch.setattr(ia, "PRED_LOG", str(tmp_path / "pred.jsonl"))
        df = _synthetic_ohlcv(n=140, seed=5)
        agent = ia.IndicatorAgent()
        agent.decide("TESTUSDT", "1h", ohlcv=df.copy(), log=False)
        assert not (tmp_path / "pred.jsonl").exists()
        agent.decide("TESTUSDT", "1h", ohlcv=df.copy(), log=True)
        assert (tmp_path / "pred.jsonl").exists()


class TestMetrics:
    def test_summarize_and_compare(self):
        from backtest.metrics import compare, summarize

        emissions = [
            {"tf": "1h", "regime": None, "reason": "nwe_direct", "action": "buy",
             "label_tb": "tp", "label_fixed": "buy", "tp_mult": 1.5, "sl_mult": 1.0,
             "fwd_return": 0.01, "atr": 1.0, "entry": 100.0}
            for _ in range(60)
        ] + [
            {"tf": "1h", "regime": None, "reason": "nwe_direct", "action": "buy",
             "label_tb": "sl", "label_fixed": "sell", "tp_mult": 1.5, "sl_mult": 1.0,
             "fwd_return": -0.01, "atr": 1.0, "entry": 100.0}
            for _ in range(40)
        ]
        s = summarize(emissions)
        g = s["groups"]["1h|all|nwe_direct"]
        assert g["n"] == 100
        assert g["tb_precision"] == pytest.approx(0.6)
        assert g["expectancy_r"] == pytest.approx(0.6 * 1.5 - 0.4 * 1.0)
        # fixed-horizon: buy prediction, 60 buy labels
        assert g["fixed_hit_rate"] == pytest.approx(0.6)

        s2 = summarize(emissions[:60])  # all tp
        cmp_ = compare(s, s2)
        assert "1h|all|nwe_direct" in cmp_["groups"]
        assert cmp_["groups"]["1h|all|nwe_direct"]["tb_precision_delta"] == pytest.approx(0.4)
