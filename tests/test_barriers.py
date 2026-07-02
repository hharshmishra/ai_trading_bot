"""Triple-barrier labeling unit tests (accuracy upgrade Phase 1)."""
import pandas as pd
import pytest

from grading.barriers import BarrierOutcome, atr_from_ohlcv, barrier_prices, triple_barrier


def _path(rows):
    """rows: list of (high, low, close)."""
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=len(rows), freq="h"),
            "open": [r[2] for r in rows],
            "high": [r[0] for r in rows],
            "low": [r[1] for r in rows],
            "close": [r[2] for r in rows],
            "volume": [1.0] * len(rows),
        }
    )


class TestBarrierPrices:
    def test_buy_direction(self):
        tp, sl = barrier_prices(entry=100.0, atr=2.0, direction="buy", tp_mult=1.5, sl_mult=1.0)
        assert tp == pytest.approx(103.0)
        assert sl == pytest.approx(98.0)

    def test_sell_direction(self):
        tp, sl = barrier_prices(entry=100.0, atr=2.0, direction="sell", tp_mult=1.5, sl_mult=1.0)
        assert tp == pytest.approx(97.0)
        assert sl == pytest.approx(102.0)

    def test_invalid_direction_raises(self):
        with pytest.raises(ValueError):
            barrier_prices(100.0, 2.0, "skip", 1.5, 1.0)


class TestTripleBarrier:
    def test_tp_first_buy(self):
        # candle 2 touches TP (103) without touching SL (98)
        path = _path([(101, 99.5, 100.5), (103.5, 100, 103), (105, 102, 104)])
        out = triple_barrier(path, entry=100.0, direction="buy", tp_price=103.0, sl_price=98.0, k=3)
        assert out.label_tb == "tp"
        assert out.hit_idx == 2
        assert out.exit_price == pytest.approx(103.0)
        assert not out.ambiguous

    def test_sl_first_buy(self):
        path = _path([(101, 99.5, 100.5), (100.5, 97.5, 98.5), (104, 98, 103.5)])
        out = triple_barrier(path, entry=100.0, direction="buy", tp_price=103.0, sl_price=98.0, k=3)
        assert out.label_tb == "sl"
        assert out.hit_idx == 2
        assert out.exit_price == pytest.approx(98.0)

    def test_tp_first_sell(self):
        # sell: tp below entry, sl above
        path = _path([(100.5, 99, 99.5), (100, 96.5, 97.5)])
        out = triple_barrier(path, entry=100.0, direction="sell", tp_price=97.0, sl_price=102.0, k=2)
        assert out.label_tb == "tp"
        assert out.hit_idx == 2

    def test_both_barriers_same_candle_is_conservative_sl(self):
        # candle 1 spans both TP and SL -> pessimistic SL-first
        path = _path([(104, 97, 100)])
        out = triple_barrier(path, entry=100.0, direction="buy", tp_price=103.0, sl_price=98.0, k=1)
        assert out.label_tb == "sl"
        assert out.ambiguous
        assert out.exit_price == pytest.approx(98.0)

    def test_timeout(self):
        path = _path([(101, 99, 100.2), (101.5, 99.5, 100.8), (102, 99, 99.9)])
        out = triple_barrier(path, entry=100.0, direction="buy", tp_price=103.0, sl_price=98.0, k=3)
        assert out.label_tb == "timeout"
        assert out.hit_idx is None
        assert out.exit_price == pytest.approx(99.9)  # close of k-th candle
        assert not out.ambiguous

    def test_incomplete_path(self):
        # only 1 candle available for k=3 and no barrier touched
        path = _path([(101, 99, 100.2)])
        out = triple_barrier(path, entry=100.0, direction="buy", tp_price=103.0, sl_price=98.0, k=3)
        assert out.label_tb == "incomplete"
        assert out.exit_price is None

    def test_hit_within_short_path_still_counts(self):
        # barrier hit on candle 1 even though path shorter than k
        path = _path([(103.5, 99, 103.2)])
        out = triple_barrier(path, entry=100.0, direction="buy", tp_price=103.0, sl_price=98.0, k=5)
        assert out.label_tb == "tp"
        assert out.hit_idx == 1

    def test_empty_path_raises(self):
        with pytest.raises(ValueError):
            triple_barrier(_path([]), entry=100.0, direction="buy",
                           tp_price=103.0, sl_price=98.0, k=3)


class TestAtr:
    def test_atr_positive_and_finite(self):
        import numpy as np
        rng = np.random.default_rng(7)
        n = 60
        close = 100 + np.cumsum(rng.normal(0, 0.5, n))
        df = pd.DataFrame({
            "timestamp": pd.date_range("2026-01-01", periods=n, freq="h"),
            "open": close, "high": close + 0.6, "low": close - 0.6,
            "close": close, "volume": np.ones(n),
        })
        atr = atr_from_ohlcv(df, period=14)
        assert atr is not None and atr > 0
