"""reset_learning.py: backs up + wipes learned state, keeps market data +
customers, leaves the DB schema intact. Runs against a tmp tree."""
from __future__ import annotations

import importlib
import json
import os
import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def tree(tmp_path, monkeypatch):
    """A fake repo tree with logs/, data/, and a seeded trading DB."""
    (tmp_path / "logs").mkdir()
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "BTCUSDT_1h.csv").write_text("keep me")
    for f in ("news_agent_policy.json", "brain_policy.json",
              "indicator_agent_policy.json", "research_agent_policy.json",
              "derivatives_agent_policy.json"):
        (tmp_path / "logs" / f).write_text(json.dumps({"weights": [[1.0]]}))
    (tmp_path / "logs" / "news_agent_policy.json.bak-5dim").write_text("{}")
    (tmp_path / "logs" / "meta_model.pkl").write_bytes(b"model")
    (tmp_path / "logs" / "predictions_log.json").write_text("{}")

    db = tmp_path / "logs" / "bitreinforcex.db"
    con = sqlite3.connect(db)
    con.executescript("""
        CREATE TABLE predictions(id TEXT); CREATE TABLE outcomes(id TEXT);
        CREATE TABLE rewards(id TEXT); CREATE TABLE sessions(id TEXT);
        CREATE TABLE news_items(id TEXT); CREATE TABLE macro_snapshots(id TEXT);
    """)
    for t in ("predictions", "outcomes", "rewards", "sessions", "news_items", "macro_snapshots"):
        con.execute(f"INSERT INTO {t} VALUES ('x')")
    con.commit(); con.close()

    subs = tmp_path / "logs" / "subscriptions.db"
    sqlite3.connect(subs).close()

    monkeypatch.chdir(tmp_path)
    import config
    # the script resolves the trading DB via os.getenv; a relative logs/ path
    # resolves under the chdir'd tmp tree, matching where the fixture wrote it
    monkeypatch.setenv("BITREINFORCEX_DB", "logs/bitreinforcex.db")
    monkeypatch.setattr(config, "MEMBERSHIP_DB", str(subs))
    monkeypatch.setattr(config, "META_MODEL_PATH", str(tmp_path / "logs" / "meta_model.pkl"))
    monkeypatch.setattr(config, "META_METRICS_PATH", str(tmp_path / "logs" / "meta_metrics.json"))
    monkeypatch.setattr(config, "CALIBRATION_PATH", str(tmp_path / "logs" / "calibration.json"))
    monkeypatch.setattr(config, "INDICATOR_CONF_PATH", str(tmp_path / "logs" / "indicator_conf.json"))
    monkeypatch.setattr(config, "ECOSYSTEMS_CACHE_PATH", str(tmp_path / "logs" / "eco.json"))
    return tmp_path, db, subs


def _run(monkeypatch, *argv):
    import scripts.reset_learning as rl
    rl = importlib.reload(rl)
    monkeypatch.setattr(rl, "_running", lambda: False)       # not "live"
    monkeypatch.setattr(sys, "argv", ["reset_learning.py", *argv])
    return rl.main()


def _counts(db, tables):
    con = sqlite3.connect(db)
    try:
        return {t: con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in tables}
    finally:
        con.close()


def test_dry_run_changes_nothing(tree, monkeypatch):
    tmp, db, _ = tree
    assert _run(monkeypatch) == 0
    assert (tmp / "logs" / "brain_policy.json").exists()      # untouched
    assert _counts(db, ["predictions"])["predictions"] == 1


def test_execute_wipes_learning_keeps_data_and_customers(tree, monkeypatch):
    tmp, db, subs = tree
    assert _run(monkeypatch, "--yes") == 0

    # policies + artifacts + line logs gone
    for f in ("brain_policy.json", "news_agent_policy.json",
              "news_agent_policy.json.bak-5dim", "meta_model.pkl", "predictions_log.json"):
        assert not (tmp / "logs" / f).exists(), f

    # learning tables emptied, schema intact; news + macro kept (no --wipe-news)
    c = _counts(db, ["predictions", "outcomes", "rewards", "sessions",
                     "news_items", "macro_snapshots"])
    assert c["predictions"] == 0 and c["rewards"] == 0 and c["sessions"] == 0
    assert c["news_items"] == 1 and c["macro_snapshots"] == 1

    # kept: market data + the customers DB
    assert (tmp / "data" / "BTCUSDT_1h.csv").read_text() == "keep me"
    assert subs.exists()

    # backup was written
    backups = list((tmp / "logs").glob("pre-reset-*"))
    assert backups and (backups[0] / "brain_policy.json").exists()


def test_wipe_news_clears_corpus(tree, monkeypatch):
    tmp, db, _ = tree
    assert _run(monkeypatch, "--yes", "--wipe-news") == 0
    assert _counts(db, ["news_items"])["news_items"] == 0


def test_refuses_when_running(tree, monkeypatch):
    import scripts.reset_learning as rl
    rl = importlib.reload(rl)
    monkeypatch.setattr(rl, "_running", lambda: True)
    monkeypatch.setattr(sys, "argv", ["reset_learning.py", "--yes"])
    assert rl.main() == 2                                     # refused, no --force
