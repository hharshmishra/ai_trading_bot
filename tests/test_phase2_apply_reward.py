"""Phase 2a verification: reward application is STATELESS (the feature-race fix).

The old bug (#4): NewsAgent/ResearchAgent stored the last prediction's features
on the shared singleton (`_last_features`/`_last_action`). Under the concurrent
48-pair batch those got clobbered, so a later learn() trained on whatever ran
last — not the graded prediction. IndicatorAgent.learn() additionally re-read the
entire multi-MB predictions log to recover the blend (#5).

These tests prove the new ``apply_reward`` updates depend ONLY on their arguments
(the recorded prediction), not on mutable instance state or the log file. So two
applications of the same payload yield identical policies even when the instance
state / prediction log differs between them.
"""
from __future__ import annotations

import copy
import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
try:  # real pandas_ta in the venv; stub if missing so import never blocks
    import pandas_ta  # noqa: F401
except Exception:  # pragma: no cover
    sys.modules.setdefault("pandas_ta", types.ModuleType("pandas_ta"))
sys.modules.setdefault("ccxt", types.ModuleType("ccxt"))


@pytest.fixture()
def tmpcwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "logs").mkdir()
    (tmp_path / "data").mkdir()
    return tmp_path


def test_news_apply_reward_ignores_instance_state(tmpcwd):
    from agents.news_agent import NewsAgent
    ag = NewsAgent()
    F, IDX, R = [0.3, -0.2, 1.0, 1.0, -1.0], 2, 1.0

    W0 = copy.deepcopy(ag._rl.policy.weights)
    ag._last_features, ag._last_action = [999.0] * 5, 0       # garbage state
    ag.apply_reward(F, IDX, R)
    W1 = copy.deepcopy(ag._rl.policy.weights)

    ag._rl.policy.weights = copy.deepcopy(W0)                  # reset, DIFFERENT garbage
    ag._last_features, ag._last_action = [-555.0] * 5, 1
    ag.apply_reward(F, IDX, R)
    W2 = copy.deepcopy(ag._rl.policy.weights)

    assert W1 == W2, "apply_reward result must not depend on _last_* instance state"
    assert W1 != W0, "apply_reward must actually update the policy"


def test_research_apply_reward_ignores_instance_state(tmpcwd):
    from agents.research_agent import ResearchAgent
    ag = ResearchAgent(prefer_csv=True)
    F = [0.1] * 10
    F[3] = 0.9
    IDX, R = 2, 1.0

    W0 = copy.deepcopy(ag._rl.policy.weights)
    ag._last_feats, ag._last_action = [999.0] * 10, 0
    ag.apply_reward(F, IDX, R)
    W1 = copy.deepcopy(ag._rl.policy.weights)

    ag._rl.policy.weights = copy.deepcopy(W0)
    ag._last_feats, ag._last_action = None, None
    ag.apply_reward(F, IDX, R)
    W2 = copy.deepcopy(ag._rl.policy.weights)

    assert W1 == W2
    assert W1 != W0


def test_indicator_apply_reward_ignores_predlog(tmpcwd):
    from agents.indicator_agent import IndicatorAgent, PRED_LOG, _load_policy, _save_policy
    ag = IndicatorAgent(prefer_csv=True)
    blend = {"type1_share": 0.8, "type2_share": 0.2, "fired_direct": "nwe"}

    W0 = copy.deepcopy(_load_policy())
    ag.apply_reward(blend, 1.0)
    W1 = copy.deepcopy(_load_policy())

    # Reset the policy and poison the predictions log with a DIFFERENT last blend.
    _save_policy(copy.deepcopy(W0))
    with open(PRED_LOG, "a") as f:
        f.write(json.dumps({"details": {"blend": {
            "type1_share": 0.05, "type2_share": 0.95, "fired_direct": "alpha_trend"}}}) + "\n")
    ag.apply_reward(blend, 1.0)
    W2 = copy.deepcopy(_load_policy())

    assert W1["weights"] == W2["weights"], "must use the passed blend, not the predictions log"
    assert W1["weights"] != W0["weights"], "must actually update the policy"
    assert "nwe" in W1["direct_signals"], "the passed blend's fired signal must be credited"
