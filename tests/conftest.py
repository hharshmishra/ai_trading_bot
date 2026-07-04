"""Suite-wide isolation: tests must never touch the real logs/ policy artifacts.

Every agent reads/writes its softmax policy through a module-level POLICY_PATH
constant, dereferenced at call time. Any test that constructs an agent without
patching the path mutates logs/*.json in the working tree (and, since the news
5->10 feature migration, would rewrite + back up the live news policy on ctor).
Redirect all of them to the test's tmp dir; agents fall back to their seeded
(Random(42)) default policies when the file does not exist. Tests that need a
specific pre-seeded policy file still monkeypatch their own path — a test-body
setattr overrides this fixture.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# MUST run before ANY project import: config.py loads .env at import time unless
# this is set, and several modules import config transitively the moment the
# first test file is collected. Setting it here (module top of the root
# conftest) guarantees the dev/CI .env never leaks flags into the suite
# (lesson 10 — "'offline' tests must be MADE offline, not assumed offline").
os.environ["BITREINFORCEX_NO_DOTENV"] = "1"

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class _NoNet:
    """requests stand-in that refuses: modules under test fall into their
    None-tolerant offline paths instead of hitting live APIs. Tests that
    exercise a transport monkeypatch .get/.post over this object."""
    def get(self, *a, **k):
        raise RuntimeError("network disabled in tests")

    def post(self, *a, **k):
        raise RuntimeError("network disabled in tests")


@pytest.fixture(autouse=True)
def _isolated_policy_paths(tmp_path, monkeypatch):
    import agents.indicator_agent as ia
    import agents.news_agent as na
    import agents.research_agent as ra
    import brain.decision_maker as dmm
    import config

    # ---- hermetic runtime (added after the phase1 flake of 2026-07-04): ----
    # macro modules fetched LIVE CoinGecko/stooq inside "offline" tests (the
    # real .env flags leak in via load_dotenv), so results drifted with the
    # actual market and their module-level TTL caches crossed test boundaries.
    from utils import macro_fetcher, macro_prices
    monkeypatch.setattr(macro_fetcher, "requests", _NoNet())
    monkeypatch.setattr(macro_fetcher, "_cache", {"fng": (0.0, None), "dom": (0.0, None)})
    monkeypatch.setattr(macro_prices, "requests", _NoNet())
    monkeypatch.setattr(macro_prices, "_cache", {})

    # get_store() singleton: without a reset, the first test to touch it pins
    # the REAL logs/bitreinforcex.db (or a stale tmp dir) for every later test.
    import persistence
    from agents import llm_client
    _default_store = persistence.Store(str(tmp_path / "default-store.db"))
    monkeypatch.setattr(persistence, "_STORE", _default_store)
    # phase1-style set_client(mock) calls leaked across tests: fresh per test
    monkeypatch.setattr(llm_client, "_ACTIVE", llm_client.LLMClient())

    monkeypatch.setattr(na, "POLICY_PATH", str(tmp_path / "news_agent_policy.json"))
    monkeypatch.setattr(na, "PREDICTIONS_LOG_PATH", str(tmp_path / "predictions_log.json"))
    monkeypatch.setattr(ra, "POLICY_PATH", str(tmp_path / "research_agent_policy.json"))
    monkeypatch.setattr(ia, "POLICY_PATH", str(tmp_path / "indicator_agent_policy.json"))
    monkeypatch.setattr(ra, "PRED_LOG", str(tmp_path / "research_predictions.jsonl"))
    monkeypatch.setattr(ia, "PRED_LOG", str(tmp_path / "indicator_predictions.jsonl"))
    monkeypatch.setattr(dmm, "POLICY_PATH", str(tmp_path / "brain_policy.json"))

    # membership store (Bot D) — separate DB, same isolation rule
    monkeypatch.setattr(config, "MEMBERSHIP_DB", str(tmp_path / "subscriptions.db"))
    # new env-driven module flags must be neutralized too (lesson 10): the flag
    # is snapshotted from os.environ at config import; a dev/CI box that runs
    # the real bot (MEMBERSHIP_ENABLED=true, ADMIN_USER_IDS set) would otherwise
    # flip flag-parity tests. Also drop the universe env so SYMBOLS stays 48.
    monkeypatch.setattr(config, "MEMBERSHIP_ENABLED", False)
    monkeypatch.setattr(config, "ADMIN_USER_IDS", frozenset())
    for _var in ("UNIVERSE_ADD", "UNIVERSE_REMOVE", "MEMBERSHIP_BOT_TOKEN",
                 "TELEGRAM_CONTROL_BOT_TOKEN", "RAZORPAY_KEY_ID", "RAZORPAY_KEY_SECRET",
                 "TRON_WALLET_ADDRESS", "TRONGRID_API_KEY"):
        monkeypatch.delenv(_var, raising=False)

    # nightly-training artifacts live under logs/ too (read via config at call time)
    for attr, name in [("INDICATOR_CONF_PATH", "indicator_conf.json"),
                       ("ECOSYSTEMS_CACHE_PATH", "ecosystems_cache.json"),
                       ("META_MODEL_PATH", "meta_model.pkl"),
                       ("META_METRICS_PATH", "meta_metrics.json"),
                       ("CALIBRATION_PATH", "calibration.json")]:
        monkeypatch.setattr(config, attr, str(tmp_path / name))

    yield
    _default_store.close()
