"""Membership P5: pro gating on the control bot + flag-off parity."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

import config
from membership.gate import requires_pro
from membership.store import DAY_S, SubsStore
from membership_fakes import mk_cmd_update

T0 = 1_800_000_000.0


@pytest.fixture
def subs(tmp_path):
    s = SubsStore(str(tmp_path / "subs.db"))
    yield s
    s.close()


def _gated(subs):
    calls = []

    async def handler(update, context):
        calls.append(update.effective_user.id)
    return requires_pro(subs, "PayBotTest")(handler), calls


def test_denied_without_subscription(subs):
    h, calls = _gated(subs)
    upd = mk_cmd_update(uid=1)
    asyncio.run(h(upd, None))
    assert calls == []
    assert "Pro plan required" in upd.message.replies[0]["text"]
    assert "PayBotTest" in upd.message.replies[0]["text"]      # deep link


def test_allowed_with_active_pro(subs):
    subs.grant(2, 30, "pro", now_ts=T0)
    h, calls = _gated(subs)
    asyncio.run(h(mk_cmd_update(uid=2), None))
    assert calls == [2]


def test_signals_only_sub_is_not_pro(subs):
    subs.grant(3, 30, "signals", now_ts=T0)
    h, calls = _gated(subs)
    upd = mk_cmd_update(uid=3)
    asyncio.run(h(upd, None))
    assert calls == []


def test_daily_cap_and_reset(subs, monkeypatch):
    monkeypatch.setattr(config, "PRO_DAILY_QUERY_CAP", 2)
    subs.grant(4, 30, "pro", now_ts=T0)
    h, calls = _gated(subs)
    asyncio.run(h(mk_cmd_update(uid=4), None))
    asyncio.run(h(mk_cmd_update(uid=4), None))
    upd = mk_cmd_update(uid=4)
    asyncio.run(h(upd, None))                                  # 3rd today
    assert calls == [4, 4]
    assert "fair-use" in upd.message.replies[0]["text"]
    # usage counter is IST-day-keyed, so the next day resets it (store-level
    # rollover proven in test_membership_store); here: cap message only once
    assert len(upd.message.replies) == 1


def test_flag_off_never_imports_membership():
    """MEMBERSHIP_ENABLED=false (the default) must leave the runtime identical
    to a build without the package: importing telegram_app pulls in NOTHING
    from membership.*."""
    assert config.MEMBERSHIP_ENABLED is False                  # default
    already = {m for m in sys.modules if m.startswith("membership")}
    # this test file imported membership itself; the parity claim is about
    # telegram_app's import graph, so check its module references instead
    import telegram_app
    import inspect
    src = inspect.getsource(telegram_app)
    assert "from membership" not in src.split("def main()")[0]  # no top-level import
    # membership imports happen only inside the MEMBERSHIP_ENABLED branch
    guarded = src.split("if _cfg.MEMBERSHIP_ENABLED and membership_token:")[1]
    assert "from membership.store import SubsStore" in guarded
