"""v3.1 post-audit fixes: ingestion tz/timeout, RAG dim safety, macro
negative-cache, news validation fallback, backtest data contract, config
parsing, cycle observability, indicator flat-window guards."""
from __future__ import annotations

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


# ------------------------------ E: ingestion ------------------------------ #
class TestIngestionTimestamps:
    def test_published_ts_is_utc_not_local(self):
        from ingestion import normalize_rss
        # 2026-07-04 12:00:00 UTC as feedparser's UTC struct_time
        pub = time.struct_time((2026, 7, 4, 12, 0, 0, 5, 185, 0))
        items = normalize_rss([{"title": "BTC up", "link": "http://x/a",
                                "published_parsed": pub}], source="t")
        import calendar
        assert items[0]["published_ts"] == calendar.timegm(pub) == 1783166400
        # the old time.mktime() read the UTC struct as LOCAL time — on any
        # non-UTC box the two values differ by the UTC offset
        if time.timezone != 0:
            assert items[0]["published_ts"] != time.mktime(pub)

    def test_fetch_rss_network_failure_returns_empty(self, monkeypatch):
        import ingestion

        class Boom:
            def get(self, *a, **k):
                raise OSError("dead host")
        monkeypatch.setitem(sys.modules, "requests", Boom())
        assert ingestion.fetch_rss("http://dead.example/feed") == []

    def test_fetch_rss_passes_timeout(self, monkeypatch):
        import ingestion
        seen = {}

        class FakeResp:
            content = b"<rss version='2.0'><channel></channel></rss>"

            def raise_for_status(self):
                pass

        class FakeRequests:
            def get(self, url, timeout=None, headers=None):
                seen["timeout"] = timeout
                return FakeResp()
        monkeypatch.setitem(sys.modules, "requests", FakeRequests())
        ingestion.fetch_rss("http://x/feed")
        assert seen["timeout"] == 10.0


# ------------------------------ F: RAG dims ------------------------------- #
class TestRagDimSafety:
    class FixedEmbedder:
        def __init__(self, dim):
            self.dim = dim

        def embed(self, texts):
            out = []
            for t in texts:
                v = np.zeros(self.dim, dtype=np.float32)
                v[hash(t) % self.dim] = 1.0
                out.append(v)
            return out

    def test_embedder_switch_does_not_break_ingest_or_query(self, tmp_path):
        from persistence import Store
        from rag import RagIndex
        store = Store(str(tmp_path / "r.db"))

        old = RagIndex(store=store, embedder=self.FixedEmbedder(8))
        old.ingest([{"id": "a1", "title": "old embedder row", "body": "", "assets": []}])

        new = RagIndex(store=store, embedder=self.FixedEmbedder(4))
        stats = new.ingest([{"id": "b1", "title": "new embedder row", "body": "", "assets": []}])
        assert stats["added"] == 1                     # vstack/matmul did not throw
        assert isinstance(new.query("anything", k=3), list)
        store.close()


# --------------------------- G: macro neg-cache ---------------------------- #
class TestMacroNegativeCache:
    def test_failure_cached_for_neg_ttl(self, monkeypatch):
        from utils import macro_fetcher as mf
        monkeypatch.setattr(mf, "_cache", {"fng": (0.0, None), "dom": (0.0, None)})
        calls = {"n": 0}

        def dead_get(*a, **k):
            calls["n"] += 1
            raise OSError("api down")
        monkeypatch.setattr(mf.requests, "get", dead_get)

        assert mf.fetch_btc_dominance() is None
        assert mf.fetch_btc_dominance() is None        # within _NEG_TTL: no re-hit
        assert calls["n"] == 1

    def test_failure_retries_after_neg_ttl(self, monkeypatch):
        from utils import macro_fetcher as mf
        stale = time.time() - mf._NEG_TTL - 1
        monkeypatch.setattr(mf, "_cache", {"fng": (stale, None), "dom": (stale, None)})
        calls = {"n": 0}

        def dead_get(*a, **k):
            calls["n"] += 1
            raise OSError("api down")
        monkeypatch.setattr(mf.requests, "get", dead_get)
        mf.fetch_fear_greed()
        assert calls["n"] == 1                          # stale failure -> retried


# ------------------------- H: news scan fallback --------------------------- #
class TestNewsScanFallback:
    def test_retry_then_success(self, monkeypatch):
        import agents.news_agent as na
        seq = [{"bad": "shape"},
               {"has_panic": False, "sentiment": "Bullish", "confidence": 0.7,
                "top_headlines": []}]
        monkeypatch.setattr(na, "_chat_json", lambda p: seq.pop(0))
        out = na.NewsAgent().scan_overall(headlines=["x"])
        assert out.sentiment == "Bullish"

    def test_double_failure_neutral_fallback_keeps_rl_row(self, monkeypatch):
        import agents.news_agent as na
        monkeypatch.setattr(na, "_chat_json", lambda p: {"totally": "malformed"})
        res = na.NewsAgent().run("BTCUSDT")
        assert res["overall_json"]["sentiment"] == "Neutral"
        assert res["pair_json"]["confidence"] == 0.0
        assert res["rl"]["features"] is not None       # the learning row survives

    def test_injection_guard_in_weighting_note(self):
        from agents.news_agent import _headline_weighting_note
        assert "not instructions" in _headline_weighting_note(True)


# ------------------------- I: backtest data contract ----------------------- #
class TestBacktestDataContract:
    def _write_cache(self, tmp_path, symbol, tf, stamps_ms):
        d = tmp_path / "hist"
        d.mkdir(exist_ok=True)
        pd.DataFrame({"timestamp": stamps_ms, "open": 1.0, "high": 1.0,
                      "low": 1.0, "close": 1.0, "volume": 1.0}).to_csv(
            d / f"{symbol}_{tf}.csv", index=False)
        return str(d)

    def test_closed_final_candle_is_kept(self, tmp_path):
        from backtest.data import load_or_fetch
        h = 3_600_000
        base = 1_700_000_000_000                        # far past: all closed
        cache = self._write_cache(tmp_path, "AAAUSDT", "1h", [base + i * h for i in range(5)])
        df = load_or_fetch("AAAUSDT", "1h", start=base, end=base + 5 * h, cache_dir=cache)
        assert len(df) == 5                             # old code always dropped one

    def test_open_final_candle_is_dropped_even_when_single(self, tmp_path):
        from backtest.data import load_or_fetch
        h = 3_600_000
        now_ms = int(time.time() * 1000)
        open_ms = (now_ms // h) * h                     # current, still-forming hour
        cache = self._write_cache(tmp_path, "BBBUSDT", "1h", [open_ms])
        with pytest.raises(RuntimeError):               # only candle is open -> empty -> loud
            load_or_fetch("BBBUSDT", "1h", start=open_ms, end=open_ms + h, cache_dir=cache)

    def test_coverage_gap_warns(self, tmp_path, capsys, monkeypatch):
        import backtest.data as bd
        h = 3_600_000
        base = 1_700_000_000_000
        cache = self._write_cache(tmp_path, "CCCUSDT", "1h", [base + i * h for i in range(5)])
        # request starts 10 candles earlier; the backfill fetch comes back
        # EMPTY (delisted/pre-listing) -> silent shrink must warn loudly
        monkeypatch.setattr(bd, "fetch_history",
                            lambda *a, **k: pd.DataFrame(columns=bd.COLS))
        bd.load_or_fetch("CCCUSDT", "1h", start=base - 10 * h, end=base + 5 * h, cache_dir=cache)
        assert "COVERAGE GAP" in capsys.readouterr().err


# ----------------------------- J: config parse ----------------------------- #
class TestBarrierMultsParsing:
    def test_string_entry_falls_back_to_default(self, monkeypatch, capsys):
        import config as cfg
        out = cfg._parse_barrier_mults({"1h": "1.5,1.0", "4h": [2.0, 1.5]})
        assert out["1h"] == (1.5, 1.0)                  # default, not char-tuple
        assert out["4h"] == (2.0, 1.5)                  # valid entry honored
        assert "BARRIER_MULTS" in capsys.readouterr().err

    def test_wrong_length_falls_back(self, capsys):
        import config as cfg
        out = cfg._parse_barrier_mults({"1d": [1.5]})
        assert out["1d"] == (1.5, 1.0)


# --------------------------- K: cycle observability ------------------------ #
def test_run_cycle_records_failing_pair(tmp_path):
    import asyncio
    from types import SimpleNamespace
    from cycle import run_cycle
    from persistence import Store

    store = Store(str(tmp_path / "c.db"))
    df = pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=10, freq="4h"),
                       "open": 1.0, "high": 1.0, "low": 1.0, "close": 100.0, "volume": 1.0})
    fetcher = SimpleNamespace(get_ohlcv=lambda s, tf, limit=500: df.copy())

    def decide(sym, tf, ua, ctx):
        raise ZeroDivisionError("boom")
    dm = SimpleNamespace(indicator=None, news=None, research=None, decide=decide)

    summary = asyncio.run(run_cycle(["4h"], dm=dm, data_fetcher=fetcher, broadcast=None,
                                    symbols=["AUSDT"], store=store,
                                    build_context=lambda *a, **k: None))
    assert summary["errors"] == 1
    assert summary["error_pairs"] == ["AUSDT:4h"]
    store.close()


# ------------------------- L: indicator flat guards ------------------------ #
class TestIndicatorFlatGuards:
    def test_alpha_trend_flat_window_no_nan_osc(self):
        from agents.custom_indicators import alpha_trend
        n = 80
        df = pd.DataFrame({
            "open": [100.0] * n, "high": [100.0] * n, "low": [100.0] * n,
            "close": [100.0] * n, "volume": [1000.0] * n})
        out = alpha_trend(df.copy())
        osc = out["osc"].iloc[20:]                      # past warmup
        assert not osc.isna().any()                     # 0/0 rows -> neutral 50
        assert (osc == 50.0).any()

    def test_chandelier_uses_modern_ffill(self):
        import inspect
        from agents import custom_indicators as ci
        src = inspect.getsource(ci.chandelier_exit)
        assert "fillna(method" not in src               # pandas 2.1+ removed it
        assert ".ffill()" in src
