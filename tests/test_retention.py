"""v3.6 retention: gc_predictions prunes only graded+old rows (with their
outcomes/rewards), shadow-log truncation stays line-aligned, preflight
membership consistency check."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config

NOW = 1_800_000_000.0
DAY = 86400.0


def _decision(pair="BTCUSDT"):
    return {"chartName": pair, "timeframe": "1h",
            "agents": {"indicator": {"action": "buy", "confidence": 0.8,
                                     "raw": {"action": "buy", "details": {"blend": {}}}}},
            "final": {"action": "buy", "confidence": 0.8, "score": 0.5},
            "policy": {"weights": {"indicator": 1.0}}}


class TestGcPredictions:
    def _store(self, tmp_path):
        from persistence import Store
        return Store(str(tmp_path / "r.db"))

    def _add(self, store, created_ts, graded):
        pid = store.record_prediction(_decision(), candle_close_ts=created_ts,
                                      entry_price=100.0, horizon_k=2,
                                      grade_due_ts=created_ts + 60,
                                      created_ts=created_ts)
        if graded:
            store.claim_grading(pid, "auto")
            store.record_reward(pid, "indicator", "buy", 1.0, "auto")
            store.mark_graded(pid, "auto")
        return pid

    def test_prunes_old_graded_keeps_rest(self, tmp_path):
        s = self._store(tmp_path)
        old_graded = self._add(s, NOW - 200 * DAY, graded=True)
        old_ungraded = self._add(s, NOW - 200 * DAY, graded=False)
        fresh_graded = self._add(s, NOW - 10 * DAY, graded=True)
        deleted = s.gc_predictions(NOW, keep_days=120)
        assert deleted == 1
        assert s.get_prediction(old_graded) is None
        assert s.get_prediction(old_ungraded) is not None     # grader still owes it
        assert s.get_prediction(fresh_graded) is not None
        assert s.rewards_for(old_graded) == []                # cascade
        s.close()

    def test_zero_keep_days_disables(self, tmp_path):
        s = self._store(tmp_path)
        self._add(s, NOW - 500 * DAY, graded=True)
        assert s.gc_predictions(NOW, keep_days=0) == 0
        s.close()


class TestShadowLogTruncate:
    def test_truncates_line_aligned(self, tmp_path, monkeypatch):
        import jobs.nightly as n
        p = tmp_path / "shadow.jsonl"
        lines = [f'{{"row": {i}}}\n' for i in range(1000)]
        p.write_text("".join(lines))
        monkeypatch.setattr(n, "_SHADOW_LOGS", (str(p),))
        n._truncate_shadow_logs(cap=4000)                     # file ~11KB > cap
        kept = p.read_text()
        assert 0 < len(kept) <= 2200                          # ~half the cap
        assert kept.startswith('{"row":')                     # line-aligned
        assert kept.endswith("\n")
        assert kept.splitlines()[-1] == '{"row": 999}'        # newest kept

    def test_small_file_untouched(self, tmp_path, monkeypatch):
        import jobs.nightly as n
        p = tmp_path / "small.jsonl"
        p.write_text('{"a":1}\n')
        monkeypatch.setattr(n, "_SHADOW_LOGS", (str(p),))
        n._truncate_shadow_logs(cap=4000)
        assert p.read_text() == '{"a":1}\n'


class TestPreflightMembership:
    def _run(self, monkeypatch, enabled, admins, token):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "pf", str(ROOT / "scripts" / "preflight.py"))
        monkeypatch.setattr(config, "MEMBERSHIP_ENABLED", enabled)
        monkeypatch.setattr(config, "ADMIN_USER_IDS", frozenset(admins))
        if token:
            monkeypatch.setenv("MEMBERSHIP_BOT_TOKEN", token)
        else:
            monkeypatch.delenv("MEMBERSHIP_BOT_TOKEN", raising=False)
        # import the module's _membership without executing the whole script
        src = (ROOT / "scripts" / "preflight.py").read_text()
        ns = {"os": __import__("os")}
        exec(src[src.index("def _membership"):src.index("def _universe")], ns)
        return ns["_membership"]

    def test_disabled_passes(self, monkeypatch):
        fn = self._run(monkeypatch, False, [], "")
        assert "disabled" in fn()

    def test_enabled_without_admins_fails(self, monkeypatch):
        fn = self._run(monkeypatch, True, [], "tok")
        with pytest.raises(RuntimeError, match="ADMIN_USER_IDS"):
            fn()

    def test_enabled_without_token_warns_not_fails(self, monkeypatch):
        fn = self._run(monkeypatch, True, [1], "")
        out = fn()
        assert "WARN" in out and "1 admin" in out
