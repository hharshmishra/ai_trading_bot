"""v3.4 extra type-2 confluence votes: unit truth tables per indicator,
fib swing confirmation / no-repaint, and agent wiring behind T2_EXTRA_VOTES
(default env must stay bit-identical to v3.3)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from agents import custom_indicators as ci
from agents.indicator_agent import IndicatorAgent


def _df(close, high=None, low=None, vol=None, opn=None, freq="h"):
    n = len(close)
    close = np.asarray(close, dtype=float)
    high = np.asarray(high, dtype=float) if high is not None else close + 0.3
    low = np.asarray(low, dtype=float) if low is not None else close - 0.3
    vol = np.asarray(vol, dtype=float) if vol is not None else np.full(n, 1000.0)
    opn = (np.asarray(opn, dtype=float) if opn is not None
           else np.concatenate([[close[0]], close[:-1]]))
    return pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=n, freq=freq),
        "open": opn, "high": high, "low": low, "close": close, "volume": vol,
    })


# ------------------------------ oscillators -------------------------------- #

class TestOscillatorVotes:
    def test_rsi30_extremes(self):
        down = np.linspace(200, 100, 80)
        up = np.linspace(100, 200, 80)
        flat = np.full(80, 100.0) + np.tile([0.1, -0.1], 40)
        assert ci.rsi30_vote(pd.Series(down)) == 1        # deeply oversold
        assert ci.rsi30_vote(pd.Series(up)) == -1         # deeply overbought
        assert ci.rsi30_vote(pd.Series(flat)) == 0

    def test_mfi_extremes(self):
        d = _df(np.linspace(200, 100, 60))
        u = _df(np.linspace(100, 200, 60))
        assert ci.mfi_vote(d["high"], d["low"], d["close"], d["volume"]) == 1
        assert ci.mfi_vote(u["high"], u["low"], u["close"], u["volume"]) == -1

    def test_cci_extremes(self):
        crash = np.concatenate([np.full(40, 100.0), np.linspace(99, 80, 5)])
        pump = np.concatenate([np.full(40, 100.0), np.linspace(101, 120, 5)])
        d, u = _df(crash), _df(pump)
        assert ci.cci_vote(d["high"], d["low"], d["close"]) == 1
        assert ci.cci_vote(u["high"], u["low"], u["close"]) == -1

    def test_votes_are_zero_on_tiny_frames(self):
        s = pd.Series([1.0, 2.0, 3.0])
        d = _df([1.0, 2.0, 3.0])
        assert ci.rsi30_vote(s) == 0
        assert ci.mfi_vote(d["high"], d["low"], d["close"], d["volume"]) == 0
        assert ci.ichimoku_vote(d["high"], d["low"], d["close"]) == 0
        assert ci.vwap_vote(d) == 0


# --------------------------------- VWAP ------------------------------------ #

class TestVwapVote:
    def test_above_both_vwaps_votes_bull(self):
        assert ci.vwap_vote(_df(np.linspace(100, 120, 100))) == 1

    def test_below_both_vwaps_votes_bear(self):
        assert ci.vwap_vote(_df(np.linspace(120, 100, 100))) == -1

    def test_daily_bars_abstain(self):
        # a UTC-day VWAP on 1d bars is the bar's own typical price — no signal
        assert ci.vwap_vote(_df(np.linspace(100, 120, 100), freq="D")) == 0


# ------------------------------- Ichimoku ---------------------------------- #

class TestIchimokuVote:
    def test_trend_agreement(self):
        up = _df(np.linspace(100, 300, 200))
        down = _df(np.linspace(300, 100, 200))
        assert ci.ichimoku_vote(up["high"], up["low"], up["close"]) == 1
        assert ci.ichimoku_vote(down["high"], down["low"], down["close"]) == -1

    def test_needs_full_cloud_history(self):
        short = _df(np.linspace(100, 150, 70))     # < 52+26 bars
        assert ci.ichimoku_vote(short["high"], short["low"], short["close"]) == 0


# ------------------------- Fibonacci golden pocket ------------------------- #

def _fib_frame(pullback_to: float, last_green: bool = True, n_pull: int = 9,
               warmup: int = 0):
    """Decline 130->100 (swing low), rally to 200 (swing high), monotonic
    pullback to ``pullback_to``; final candle colored via ``last_green``.
    ``warmup`` prepends flat bars (no pivots) so the full agent path — which
    drops MA50/StochRSI warmup rows — still sees the same swing structure."""
    seg1 = np.linspace(130, 100, 10)                     # swing low at 100
    seg2 = np.linspace(103, 200, 30)                     # rally, high 200
    seg3 = np.linspace(197, pullback_to, n_pull)         # pullback
    close = np.concatenate([np.full(warmup, 130.0), seg1, seg2, seg3])
    opn = np.concatenate([[close[0]], close[:-1]]).copy()
    if last_green:
        opn[-1] = close[-1] - 1.0                        # close > open: rejection up
    else:
        opn[-1] = close[-1] + 1.0
    return _df(close, high=close + 0.5, low=close - 0.5, opn=opn)


class TestFib:
    def test_swings_confirm_late_and_never_repaint(self):
        df = _fib_frame(146)
        peak_bar = 39                                     # index of the 200 high
        upto_unconfirmed = df.iloc[: peak_bar + 3]        # only 2 bars after peak
        upto_confirmed = df.iloc[: peak_bar + 4]          # w=3 bars after peak
        hi_prices = lambda d: [p for _, p in ci.confirmed_swings(d["high"], d["low"])[0]]
        assert 200.5 not in hi_prices(upto_unconfirmed)   # not yet a pivot
        assert 200.5 in hi_prices(upto_confirmed)         # confirms w bars late
        # append future bars: the confirmed level must NOT move (no repaint)
        assert 200.5 in hi_prices(df)

    def test_golden_pocket_rejection_votes_bull(self):
        out = ci.fib_confluence_vote(_fib_frame(146))     # 0.54 retrace of 100->200
        assert out["vote"] == 1 and out["leg"] == "up"
        assert 0.45 <= out["ratio"] <= 0.65

    def test_far_from_pocket_no_vote(self):
        assert ci.fib_confluence_vote(_fib_frame(185))["vote"] == 0   # 0.15 retrace

    def test_red_candle_in_pocket_no_vote(self):
        assert ci.fib_confluence_vote(_fib_frame(146, last_green=False))["vote"] == 0

    def test_down_leg_mirrors(self):
        df = _fib_frame(146)
        m = df.copy()                                     # price-mirror the frame
        for c in ("open", "high", "low", "close"):
            m[c] = 300.0 - df[c]
        m[["high", "low"]] = m[["low", "high"]].to_numpy()   # re-order after mirror
        out = ci.fib_confluence_vote(m)
        assert out["vote"] == -1 and out["leg"] == "down"


# ------------------------------ agent wiring -------------------------------- #

def _agent_frame(n=220):
    rng = np.random.default_rng(7)
    close = 100 + np.cumsum(rng.normal(0.05, 0.6, n))
    return _df(close, high=close + 0.8, low=close - 0.8)


class TestAgentWiring:
    def test_enabled_vote_lands_in_extras(self, monkeypatch):
        monkeypatch.setattr(config, "T2_EXTRA_VOTES", frozenset({"mfi", "vwap"}))
        d = IndicatorAgent().decide("BTCUSDT", "1h", ohlcv=_agent_frame(), log=False)
        extras = d.details["type2"]["extras"]
        assert set(extras) == {"mfi", "vwap"}
        assert all(v in (-1, 0, 1) for v in extras.values())

    def test_fib_ratio_exposed_when_vote_fires(self, monkeypatch):
        monkeypatch.setattr(config, "T2_EXTRA_VOTES", frozenset({"fib"}))
        d = IndicatorAgent().decide("BTCUSDT", "1h",
                                    ohlcv=_fib_frame(146, warmup=120), log=False)
        extras = d.details["type2"]["extras"]
        assert extras["fib"] == 1 and "fib_ratio" in extras

    def test_default_env_output_is_unchanged(self):
        # conftest pins T2_EXTRA_VOTES = frozenset(): v3.3 regression guard —
        # no extras key, no v_ columns, tally untouched.
        d = IndicatorAgent().decide("BTCUSDT", "1h", ohlcv=_agent_frame(), log=False)
        assert "extras" not in d.details["type2"]

    def test_extra_vote_moves_the_tally(self, monkeypatch):
        frame = _df(np.linspace(200, 100, 220))           # hard downtrend
        base = IndicatorAgent().decide("X", "1h", ohlcv=frame, log=False)
        monkeypatch.setattr(config, "T2_EXTRA_VOTES", frozenset({"rsi30"}))
        with_vote = IndicatorAgent().decide("X", "1h", ohlcv=frame, log=False)
        b0 = base.details["type2"]["votes"]
        b1 = with_vote.details["type2"]["votes"]
        # oversold in a downtrend: rsi30 adds exactly one bull vote
        assert b1["bull"] == b0["bull"] + 1 and b1["bear"] == b0["bear"]
