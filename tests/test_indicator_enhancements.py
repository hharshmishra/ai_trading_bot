"""Phase D: pivot divergences, divergence votes, empirical-Bayes direct conf."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config


def _div_frame(bullish=True, n=80):
    """Price makes two pivots (second lower low for bullish case) while the
    oscillator makes a higher low — classic divergence geometry."""
    rng = np.random.default_rng(5)
    base = 100 + rng.normal(0, 0.05, n)
    price = base.copy()
    if bullish:
        price[40] = 95.0     # pivot low 1
        price[70] = 93.5     # pivot low 2 (lower low), confirms at 73
    else:
        price[40] = 105.0
        price[70] = 106.5    # higher high
    ts = pd.date_range("2024-01-01", periods=n, freq="4h")
    return pd.DataFrame({"timestamp": ts, "open": price, "high": price + 0.3,
                         "low": price - 0.3, "close": price,
                         "volume": np.ones(n) * 100})


class TestPivotDivergence:
    def test_bullish_divergence(self):
        from agents.custom_indicators import pivot_divergence
        df = _div_frame(bullish=True)
        # oscillator: same pivots but HIGHER second low
        osc = pd.Series(np.full(len(df), 50.0))
        osc.iloc[40] = 20.0
        osc.iloc[70] = 35.0
        assert pivot_divergence(df["close"], osc) == 1

    def test_bearish_divergence(self):
        from agents.custom_indicators import pivot_divergence
        df = _div_frame(bullish=False)
        osc = pd.Series(np.full(len(df), 50.0))
        osc.iloc[40] = 80.0
        osc.iloc[70] = 65.0   # lower high vs price higher high
        assert pivot_divergence(df["close"], osc) == -1

    def test_no_divergence_when_confirming(self):
        from agents.custom_indicators import pivot_divergence
        df = _div_frame(bullish=True)
        osc = pd.Series(np.full(len(df), 50.0))
        osc.iloc[40] = 35.0
        osc.iloc[70] = 20.0   # oscillator confirms the lower low
        assert pivot_divergence(df["close"], osc) == 0

    def test_stale_divergence_ignored(self):
        from agents.custom_indicators import pivot_divergence
        n = 120
        rng = np.random.default_rng(6)
        price = pd.Series(100 + rng.normal(0, 0.05, n))
        price.iloc[20] = 95.0
        price.iloc[40] = 93.5     # divergence confirmed ~77 bars ago
        osc = pd.Series(np.full(n, 50.0))
        osc.iloc[20] = 20.0
        osc.iloc[40] = 35.0
        assert pivot_divergence(price, osc, recency=12) == 0

    def test_no_lookahead(self):
        """The pivot at bar t needs pivot_w bars AFTER it — a divergence must
        not be visible until those bars exist."""
        from agents.custom_indicators import pivot_divergence
        df = _div_frame(bullish=True)
        osc = pd.Series(np.full(len(df), 50.0))
        osc.iloc[40] = 20.0
        osc.iloc[70] = 35.0
        # cut the frame at bar 71: second pivot (70) not yet confirmed (needs 73)
        assert pivot_divergence(df["close"].iloc[:72], osc.iloc[:72]) == 0


class TestDivergenceVotes:
    def _df(self):
        rng = np.random.default_rng(9)
        n = 160
        close = 100 + np.cumsum(rng.normal(0, 0.4, n))
        ts = pd.date_range("2024-01-01", periods=n, freq="4h")
        return pd.DataFrame({"timestamp": ts, "open": close, "high": close + 0.5,
                             "low": close - 0.5, "close": close,
                             "volume": np.abs(rng.normal(1000, 100, n))})

    def test_columns_only_when_flag_and_tf(self, monkeypatch):
        import agents.indicator_agent as ia
        agent = ia.IndicatorAgent()

        monkeypatch.setattr(config, "DIVERGENCE_VOTES", True)
        raw_4h = agent._compute_raw_indicators(self._df(), "4h")
        assert "rsi_div" in raw_4h.columns and "obv_div" in raw_4h.columns
        assert not raw_4h["rsi_div"].isna().any()   # 0-filled, never NaN

        raw_1h = agent._compute_raw_indicators(self._df(), "1h")
        assert "rsi_div" not in raw_1h.columns      # 4h/1d only

        monkeypatch.setattr(config, "DIVERGENCE_VOTES", False)
        raw_off = agent._compute_raw_indicators(self._df(), "4h")
        assert "rsi_div" not in raw_off.columns     # flag-off parity

    def test_votes_counted(self, monkeypatch):
        import agents.indicator_agent as ia
        monkeypatch.setattr(config, "DIVERGENCE_VOTES", True)
        agent = ia.IndicatorAgent()
        raw = agent._compute_raw_indicators(self._df(), "4h")
        raw["rsi_div"] = 1.0    # force a bullish divergence reading
        raw["obv_div"] = 0.0
        with_div = agent._type2_rules(raw)
        raw["rsi_div"] = 0.0
        without = agent._type2_rules(raw)
        assert (with_div["votes"]["bull"] - without["votes"]["bull"]) == 1


class TestEmpiricalDirectConf:
    def _rows(self, name, wins, losses):
        rows = []
        for w in [1] * wins + [0] * losses:
            rows.append({"indicator_action": "buy",
                         "realized_label": "buy" if w else "sell",
                         "indicator_blend": {"fired_direct": name}})
        return rows

    def test_fit_and_read(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "INDICATOR_CONF_PATH", str(tmp_path / "ic.json"))
        from jobs.nightly import fit_direct_conf
        rows = self._rows("chandelier_exit", 30, 20) + self._rows("nwe", 10, 5)
        payload = fit_direct_conf(rows, m=20, min_n=30)
        assert "chandelier_exit" in payload["conf"]
        assert "nwe" not in payload["conf"]          # n=15 < min_n
        ce = payload["conf"]["chandelier_exit"]
        # shrunk toward global mean, conf in [0.5, 0.95]
        assert 0.5 <= ce["conf"] <= 0.95

        import agents.indicator_agent as ia
        ia._DIRECT_CONF_CACHE.update({"mtime": None, "conf": {}})
        monkeypatch.setattr(config, "EMPIRICAL_DIRECT_CONF", True)
        assert ia._direct_conf("chandelier_exit", 0.9) == pytest.approx(ce["conf"])
        assert ia._direct_conf("alpha_trend", 0.9) == 0.9   # no entry -> default

        monkeypatch.setattr(config, "EMPIRICAL_DIRECT_CONF", False)
        assert ia._direct_conf("chandelier_exit", 0.9) == 0.9  # flag off -> default

    def test_shrinkage_math(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "INDICATOR_CONF_PATH", str(tmp_path / "ic.json"))
        from jobs.nightly import fit_direct_conf
        rows = self._rows("x", 40, 0)    # perfect record, n=40
        payload = fit_direct_conf(rows, m=20, min_n=30)
        # global p̄ = 1.0 here, so shrunk stays 1.0 and conf = 0.95
        assert payload["conf"]["x"]["conf"] == pytest.approx(0.95)
