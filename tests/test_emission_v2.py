"""v3.8 edge-first emission: evidence ledger + gate v3 + cycle wiring.

The 21-day audit showed the hand-flag gate stack suppressing its only proven
signal (NWE crossings, 40-50% outside calm vol) down to ~1.5 emissions/day.
The ledger replaces per-source flags with per-cohort measured precision.
"""
import asyncio
import json
from types import SimpleNamespace

import pandas as pd
import pytest

import config
from jobs.ledger import (build_ledger, ledger_verdict, load_ledger_cached,
                         regime_group, save_ledger, vol_band, wilson_lb)
from tests.test_phase4_runtime import make_decision


def _rows(source, tf, regime, vol, n, hits, emitted=0):
    out = []
    for i in range(n):
        out.append({"candidate_trigger": source, "candidate_action": "buy",
                    "realized_label": "buy" if i < hits else "sell",
                    "tf": tf, "regime": regime,
                    "regime_feats": {"vol_pct": vol},
                    "emitted": 1 if i < emitted else 0})
    return out


class TestLedgerMath:
    def test_wilson_lb_basics(self):
        assert wilson_lb(0, 0) == 0.0
        assert wilson_lb(50, 100) == pytest.approx(0.4038, abs=1e-3)
        assert wilson_lb(55, 129) == pytest.approx(0.3438, abs=1e-3)  # audit NWE

    def test_bands_and_groups(self):
        assert vol_band(0.1) == "calm" and vol_band(0.5) == "normal"
        assert vol_band(0.9) == "elevated" and vol_band(None) == "unknown"
        assert regime_group("trend_up") == "trending"
        assert regime_group("ranging") == "ranging"
        assert regime_group(None) == "mixed"

    def test_build_counts_cohort_and_global(self):
        led = build_ledger(_rows("nwe", "1h", "mixed", 0.8, 40, 18, emitted=7))
        c = led["cohorts"]["nwe|1h|mixed|elevated"]
        g = led["cohorts"]["nwe|1h|*|*"]
        assert c["n"] == 40 and c["hit"] == 18 and g["n"] == 40
        assert c["rate"] == pytest.approx(0.45)
        assert led["emitted_by_source"]["nwe"] == 7

    def test_build_skips_non_candidates(self):
        led = build_ledger([{"candidate_trigger": None, "candidate_action": "buy",
                             "realized_label": "buy", "tf": "1h"},
                            {"candidate_trigger": "nwe", "candidate_action": "skip",
                             "realized_label": "buy", "tf": "1h"}])
        assert led["cohorts"] == {}


class TestLedgerVerdict:
    def _led(self):
        return build_ledger(
            _rows("nwe", "1h", "mixed", 0.8, 60, 27)        # 45% @ 60, LB .327
            + _rows("nwe", "1h", "mixed", 0.5, 30, 12)      # 40% @ 30, LB .246
            + _rows("conf", "4h", "trend_up", 0.5, 80, 7))  # 8.8%

    def test_cohort_passes_two_part_test(self):
        ok, why, st = ledger_verdict(self._led(), "nwe", "1h", "mixed", 0.8)
        assert ok and why == "ledger_ok" and st["n"] == 60

    def test_cohort_rate_ok_but_lb_guard_fails(self):
        ok, why, st = ledger_verdict(self._led(), "nwe", "1h", "mixed", 0.5)
        assert not ok and why == "ledger_below_floor"

    def test_junk_source_dead(self):
        ok, why, _ = ledger_verdict(self._led(), "conf", "4h", "trend_up", 0.5)
        assert not ok and why == "ledger_below_floor"

    def test_unmeasured_cohort_falls_to_global(self):
        # nwe|1h ranging/calm has no cohort -> global (90 rows, 43.3%) decides
        ok, why, st = ledger_verdict(self._led(), "nwe", "1h", "ranging", 0.1)
        assert ok and why == "ledger_ok" and st["probation"] == "global"

    def test_new_source_probation_budget(self, monkeypatch):
        led = self._led()
        led["emitted_by_source"]["sms_bos"] = 0
        ok, why, _ = ledger_verdict(led, "sms_bos", "1h", "mixed", 0.5)
        assert ok and why == "ledger_probation"
        led["emitted_by_source"]["sms_bos"] = config.LEDGER_PROBATION_N
        ok, why, _ = ledger_verdict(led, "sms_bos", "1h", "mixed", 0.5)
        assert not ok and why == "ledger_cold"

    def test_missing_ledger(self):
        ok, why, _ = ledger_verdict(None, "nwe", "1h", "mixed", 0.5)
        assert not ok and why == "ledger_missing"


class TestLedgerArtifact:
    def test_save_load_cache(self, tmp_path, monkeypatch):
        path = str(tmp_path / "ledger.json")
        monkeypatch.setattr(config, "LEDGER_PATH", path)
        import jobs.ledger as jl
        monkeypatch.setattr(jl, "_CACHE", {"path_mtime": None, "ledger": None})
        assert load_ledger_cached() is None
        save_ledger(build_ledger(_rows("nwe", "1h", "mixed", 0.8, 30, 14)))
        led = load_ledger_cached()
        assert led and "nwe|1h|mixed|elevated" in led["cohorts"]

    def test_nightly_builds_ledger(self, tmp_path, monkeypatch):
        from jobs.nightly import run_nightly_training
        from persistence import Store
        monkeypatch.setattr(config, "LEDGER_PATH", str(tmp_path / "led.json"))
        store = Store(str(tmp_path / "n.db"))
        summary = run_nightly_training(store)
        assert summary["ledger_cohorts"] == 0          # empty store, still writes
        assert json.load(open(config.LEDGER_PATH))["cohorts"] == {}
        store.close()


def _decision(nwe=None, sms=None, conf=0.1, action="skip", regime="mixed",
              vol_pct=0.8):
    dec = make_decision(pair="AUSDT", tf="1h", action=action, conf=conf, nwe=nwe)
    det = dec["agents"]["indicator"]["raw"]["details"]
    if sms:
        det["direct_signals"].append({"name": sms[0], "signal": sms[1],
                                      "confidence": 0.6})
    det["regime"] = regime
    det["regime_feats"] = {"vol_ok": True, "vol_pct": vol_pct, "atr": 1.0}
    return dec


class TestGateV3:
    @pytest.fixture(autouse=True)
    def _ledger(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "LEDGER_PATH", str(tmp_path / "led.json"))
        import jobs.ledger as jl
        monkeypatch.setattr(jl, "_CACHE", {"path_mtime": None, "ledger": None})
        save_ledger(build_ledger(
            _rows("nwe", "1h", "mixed", 0.8, 60, 27)
            + _rows("conf", "1h", "mixed", 0.8, 80, 8)))

    def test_proven_source_emits_with_its_own_direction(self):
        from signals import should_emit_signal_v3
        res = _decision(nwe="sell", action="buy", conf=0.9)
        emit, action, nwe, conf, reason, ct, ca, stats = should_emit_signal_v3(res)
        assert emit and action == "sell" == ca and ct == "nwe" == reason
        assert stats["n"] == 60

    def test_junk_source_suppressed_with_reason(self):
        from signals import should_emit_signal_v3
        res = _decision(action="buy", conf=0.9)          # conf candidate only
        emit, action, nwe, conf, reason, ct, ca, stats = should_emit_signal_v3(res)
        assert not emit and reason == "ledger_below_floor"
        assert ct == "conf" and ca == "buy"

    def test_no_candidates_records_nothing(self):
        from signals import should_emit_signal_v3
        emit, action, nwe, conf, reason, ct, ca, stats = should_emit_signal_v3(
            _decision())
        assert not emit and reason == "" and ct is None and ca is None

    def test_sms_shadow_never_emits_and_never_steals(self, monkeypatch):
        from signals import should_emit_signal_v3
        monkeypatch.setattr(config, "SMS_EMIT", False)
        # sms alone -> recorded, not emitted
        res = _decision(sms=("sms_bos", "buy"))
        emit, action, nwe, conf, reason, ct, ca, stats = should_emit_signal_v3(res)
        assert not emit and reason == "sms_shadow" and ct == "sms_bos" and ca == "buy"
        # sms + proven nwe in the same cycle -> nwe emits, slot not stolen
        res = _decision(nwe="sell", sms=("sms_bos", "buy"))
        emit, action, *_r = should_emit_signal_v3(res)
        assert emit and action == "sell" and _r[2] == "nwe"

    def test_sms_emit_true_uses_probation(self, monkeypatch):
        from signals import should_emit_signal_v3
        monkeypatch.setattr(config, "SMS_EMIT", True)
        res = _decision(sms=("sms_bos", "buy"))
        emit, action, nwe, conf, reason, ct, ca, stats = should_emit_signal_v3(res)
        assert emit and reason == "sms_bos" and stats["probation"] == "new_source"

    def test_measured_evidence_outranks_probation(self, monkeypatch):
        from signals import should_emit_signal_v3
        monkeypatch.setattr(config, "SMS_EMIT", True)
        res = _decision(nwe="sell", sms=("sms_bos", "buy"))
        emit, action, *_r = should_emit_signal_v3(res)
        assert emit and action == "sell" and _r[2] == "nwe"

    def test_missing_ledger_file_suppresses(self, monkeypatch):
        import os
        os.remove(config.LEDGER_PATH)
        import jobs.ledger as jl
        monkeypatch.setattr(jl, "_CACHE", {"path_mtime": None, "ledger": None})
        from signals import should_emit_signal_v3
        emit, _, _, _, reason, ct, _, _ = should_emit_signal_v3(_decision(nwe="buy"))
        assert not emit and reason == "ledger_missing" and ct == "nwe"


class TestCycleV3:
    def test_emitted_row_carries_candidate_and_ledger_note(self, tmp_path, monkeypatch):
        from cycle import run_cycle
        from persistence import Store
        monkeypatch.setattr(config, "EMISSION_V2_ENABLED", True)
        monkeypatch.setattr(config, "META_GATE_ENABLED", False)
        monkeypatch.setattr(config, "LEDGER_PATH", str(tmp_path / "led.json"))
        import jobs.ledger as jl
        monkeypatch.setattr(jl, "_CACHE", {"path_mtime": None, "ledger": None})
        save_ledger(build_ledger(_rows("nwe", "1h", "mixed", 0.8, 60, 27)))

        store = Store(str(tmp_path / "c.db"))
        df = pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=10, freq="1h"),
                           "open": 1.0, "high": 1.0, "low": 1.0, "close": 100.0,
                           "volume": 1.0})
        fetcher = SimpleNamespace(get_ohlcv=lambda s, tf, limit=500: df.copy())
        dec = _decision(nwe="sell", action="buy", conf=0.4)
        dm = SimpleNamespace(indicator=None, news=None, research=None,
                             decide=lambda sym, tf, ua, ctx: dec)
        sent = []

        async def fake_broadcast(**kw):
            sent.append(kw)
            return "sess"

        summary = asyncio.run(run_cycle(
            ["1h"], dm=dm, data_fetcher=fetcher, broadcast=fake_broadcast,
            symbols=["AUSDT"], store=store, build_context=lambda *a, **k: None))
        assert summary["emitted"] == 1
        assert sent[0]["overall"] == "sell" and sent[0]["reason"] == "nwe"
        assert (sent[0]["decision"].get("meta") or {}).get("ledger", {}).get("n") == 60
        with store._lock:
            row = store.conn.execute(
                "SELECT candidate_trigger, candidate_action, trigger_source, "
                "gate_reason, tp_price, sl_price, final_action FROM predictions").fetchone()
        assert row["candidate_trigger"] == "nwe"
        assert row["candidate_action"] == "sell"
        assert row["trigger_source"] == "nwe" and row["gate_reason"] == "nwe"
        # barriers anchored on the SENT sell (tp below entry), not the buy final
        assert row["final_action"] == "buy"
        assert row["tp_price"] < 100.0 < row["sl_price"]
        store.close()

    def test_flag_off_is_v371_path(self, tmp_path, monkeypatch):
        """EMISSION_V2_ENABLED=false must leave the v2 gate outcome intact."""
        from cycle import run_cycle
        from persistence import Store
        monkeypatch.setattr(config, "EMISSION_V2_ENABLED", False)
        monkeypatch.setattr(config, "GATE_V2_ENABLED", True)
        monkeypatch.setattr(config, "META_GATE_ENABLED", False)
        store = Store(str(tmp_path / "c2.db"))
        df = pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=10, freq="1h"),
                           "open": 1.0, "high": 1.0, "low": 1.0, "close": 100.0,
                           "volume": 1.0})
        fetcher = SimpleNamespace(get_ohlcv=lambda s, tf, limit=500: df.copy())
        dec = _decision(nwe="sell", regime="ranging", vol_pct=0.5)
        dm = SimpleNamespace(indicator=None, news=None, research=None,
                             decide=lambda sym, tf, ua, ctx: dec)
        sent = []

        async def fake_broadcast(**kw):
            sent.append(kw)
            return "sess"

        summary = asyncio.run(run_cycle(
            ["1h"], dm=dm, data_fetcher=fetcher, broadcast=fake_broadcast,
            symbols=["AUSDT"], store=store, build_context=lambda *a, **k: None))
        assert summary["emitted"] == 1
        assert sent[0]["reason"] == "nwe_ranging"       # v2 vocabulary intact
        store.close()


class TestMessageNotes:
    def test_ledger_note(self):
        from signals import ledger_note_from
        assert ledger_note_from({"source": "nwe", "rate": 0.45, "n": 60,
                                 "lb": 0.327}) == "nwe: 45% hit (n=60, LB 33%)"
        assert "probation" in ledger_note_from({"source": "sms_bos",
                                                "probation": "new_source"})
        assert ledger_note_from(None) is None

    def test_sms_note(self):
        from signals import sms_note_from
        note = sms_note_from({"sms": {"trend": {"1h": 1, "4h": -1, "1d": 0},
                                      "strength": -33.3, "confidence": 60.0}})
        assert "1h ▲" in note and "4h ▼" in note and "1d ━" in note
        assert "strength -33" in note and "conf 60%" in note
        assert sms_note_from({}) is None

    def test_fmt_message_includes_context(self):
        from signals import fmt_signal_message
        text = fmt_signal_message(
            "BTCUSDT", "1h", "sell", "sell", 0.4, "nwe",
            regime="mixed", meta_p=0.62,
            ledger_note="nwe: 45% hit (n=60, LB 33%)",
            sms_note="1h ▲ 4h ▼ 1d ━ · strength -33 · conf 60%")
        assert "MODEL P(CORRECT):</b> 62%" in text
        assert "TRACK RECORD:</b> nwe: 45% hit" in text
        assert "MARKET STRUCTURE:</b> 1h ▲" in text
        assert "NWE crossing (evidence-gated)" in text
