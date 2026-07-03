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

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def _isolated_policy_paths(tmp_path, monkeypatch):
    import agents.indicator_agent as ia
    import agents.news_agent as na
    import agents.research_agent as ra
    import brain.decision_maker as dmm
    import config

    monkeypatch.setattr(na, "POLICY_PATH", str(tmp_path / "news_agent_policy.json"))
    monkeypatch.setattr(na, "PREDICTIONS_LOG_PATH", str(tmp_path / "predictions_log.json"))
    monkeypatch.setattr(ra, "POLICY_PATH", str(tmp_path / "research_agent_policy.json"))
    monkeypatch.setattr(ia, "POLICY_PATH", str(tmp_path / "indicator_agent_policy.json"))
    monkeypatch.setattr(dmm, "POLICY_PATH", str(tmp_path / "brain_policy.json"))

    # nightly-training artifacts live under logs/ too (read via config at call time)
    for attr, name in [("INDICATOR_CONF_PATH", "indicator_conf.json"),
                       ("ECOSYSTEMS_CACHE_PATH", "ecosystems_cache.json"),
                       ("META_MODEL_PATH", "meta_model.pkl"),
                       ("META_METRICS_PATH", "meta_metrics.json"),
                       ("CALIBRATION_PATH", "calibration.json")]:
        monkeypatch.setattr(config, attr, str(tmp_path / name))
