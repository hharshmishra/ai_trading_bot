"""Auto-labeling RL grader (Phase 3) — the system trains itself on realized price.

For each prediction whose grade-due time has passed, fetch the candle that closed
``k`` periods after the signal, compute the realized direction (forward return vs
a per-timeframe threshold), reward/punish each agent against its OWN recorded
prediction (no instance state — uses the SQLite snapshot), and persist the
outcome. No human in the loop.

Manual Telegram feedback overrides auto (see ``apply_manual_feedback``): the
grader only touches ``label_source='pending'`` rows, and if a human disagrees
with an already-auto-graded prediction we apply a correction = manual - auto so
the net effect on the policy equals the human's verdict.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from persistence import Store, get_store

# Per-timeframe grading config. Tunable — backtest against history in a later pass.
HORIZON_K = {"1h": 3, "4h": 2, "1d": 1, "1w": 1}          # candles ahead to look
THRESHOLD = {"1h": 0.004, "4h": 0.010, "1d": 0.025, "1w": 0.05}  # |fwd return| band

REWARD_CORRECT = 1.0
REWARD_WRONG = -4.0


def _norm_action(a: Optional[str]) -> str:
    a = (a or "").strip().lower()
    if a in ("buy", "b", "bull", "bullish"):
        return "buy"
    if a in ("sell", "s", "bear", "bearish"):
        return "sell"
    return "skip"


def realized_label(forward_return: float, tf: str) -> str:
    th = THRESHOLD.get(tf, 0.01)
    if forward_return >= th:
        return "buy"
    if forward_return <= -th:
        return "sell"
    return "skip"          # move stayed in-band -> 'skip' was the right call


def reward_for(predicted: Optional[str], realized: str,
               correct: float = REWARD_CORRECT, wrong: float = REWARD_WRONG) -> float:
    return correct if _norm_action(predicted) == realized else wrong


class Grader:
    def __init__(self, decision_maker, data_fetcher=None, store: Optional[Store] = None):
        self.dm = decision_maker
        self.store = store or get_store()
        # Reuse the brain's data fetcher (and thus the per-cycle OHLCV cache) if
        # none is supplied.
        self.data = data_fetcher or getattr(getattr(decision_maker, "indicator", None), "data", None)

    # ------------------------------------------------------------------ #
    # realized price
    # ------------------------------------------------------------------ #
    def _realized_close(self, pair: str, tf: str, candle_close_ts: Optional[float], k: int) -> Optional[float]:
        """Close of the k-th candle strictly after candle_close_ts, or None if it
        has not printed yet."""
        if self.data is None or candle_close_ts is None:
            return None
        try:
            df = self.data.get_ohlcv(pair, tf, limit=max(50, k + 5))
        except Exception:
            return None
        if df is None or getattr(df, "empty", True) or "timestamp" not in df.columns:
            return None
        ts = pd.to_datetime(df["timestamp"])
        close_dt = pd.to_datetime(candle_close_ts, unit="s")
        after = df[ts > close_dt]
        if len(after) < k:
            return None
        return float(after.iloc[k - 1]["close"])

    # ------------------------------------------------------------------ #
    # auto grading
    # ------------------------------------------------------------------ #
    def grade_once(self, now_ts: Optional[float] = None) -> List[Dict[str, Any]]:
        """Grade every due prediction whose horizon has printed. Returns the
        list of graded results (skips ones still waiting for candles)."""
        graded = []
        for p in self.store.get_due_predictions(now_ts):
            res = self._grade_prediction(p)
            if res is not None:
                graded.append(res)
        return graded

    def _grade_prediction(self, p: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        tf = p["tf"]
        k = HORIZON_K.get(tf, 1)
        th = THRESHOLD.get(tf, 0.01)
        entry = p.get("entry_price")
        if not entry:
            return None
        realized_close = self._realized_close(p["pair"], tf, p.get("candle_close_ts"), k)
        if realized_close is None:
            return None  # horizon not reached yet; stays pending for a later pass
        fr = (realized_close - entry) / entry
        label = realized_label(fr, tf)
        rewards = self._apply_rewards(p, label, source="auto")
        self.store.record_outcome(p["id"], fr, label, th, k, source="auto")
        self.store.mark_graded(p["id"], "auto")
        return {"id": p["id"], "pair": p["pair"], "tf": tf, "forward_return": fr,
                "realized_label": label, "rewards": rewards}

    def _apply_rewards(self, p: Dict[str, Any], label: str, source: str,
                       news_reward_override: Optional[float] = None) -> Dict[str, float]:
        """Apply per-agent rewards from the STORED payloads. News may take an
        explicit numeric reward (the Telegram 1.0/-4.0); the others match against
        the realized label."""
        rewards: Dict[str, float] = {}

        if p.get("news_feats") is not None and p.get("news_action_idx") is not None:
            rn = news_reward_override if news_reward_override is not None else reward_for(p["news_action"], label)
            self.dm.news.apply_reward(p["news_feats"], p["news_action_idx"], rn)
            self.store.record_reward(p["id"], "news", p["news_action"], rn, source)
            rewards["news"] = rn

        if p.get("research_feats") is not None and p.get("research_action_idx") is not None:
            rr = reward_for(p["research_action"], label)
            self.dm.research.apply_reward(p["research_feats"], p["research_action_idx"], rr)
            self.store.record_reward(p["id"], "research", p["research_action"], rr, source)
            rewards["research"] = rr

        if p.get("indicator_blend") is not None:
            ri = reward_for(p["indicator_action"], label)
            self.dm.indicator.apply_reward(p["indicator_blend"], ri)
            self.store.record_reward(p["id"], "indicator", p["indicator_action"], ri, source)
            rewards["indicator"] = ri

        # Brain priority weights learn from the same ground truth.
        agent_results = {
            "news": {"action": _norm_action(p.get("news_action")), "confidence": p.get("news_conf") or 0.0},
            "research": {"action": _norm_action(p.get("research_action")), "confidence": p.get("research_conf") or 0.0},
            "indicator": {"action": _norm_action(p.get("indicator_action")), "confidence": p.get("indicator_conf") or 0.0},
        }
        news_reward = rewards.get("news", reward_for(p.get("news_action"), label))
        try:
            self.dm.apply_brain_feedback(agent_results, label, news_reward)
        except Exception:
            pass
        return rewards

    # ------------------------------------------------------------------ #
    # manual feedback (Telegram) — precedence over auto
    # ------------------------------------------------------------------ #
    def apply_manual_feedback(self, prediction_id: str, true_outcome: str,
                              news_reward: Optional[float] = None) -> Dict[str, Any]:
        p = self.store.get_prediction(prediction_id)
        if p is None:
            return {"status": "unknown_prediction"}
        label = _norm_action(true_outcome)
        prior = p.get("label_source")

        if prior == "manual":
            return {"status": "already_manual"}

        if prior == "auto":
            # Already auto-graded; net the policy to the human verdict.
            return self._apply_correction(p, label, news_reward)

        # pending -> apply the manual labels directly
        rewards = self._apply_rewards(p, label, source="manual", news_reward_override=news_reward)
        self.store.record_outcome(p["id"], None, label, THRESHOLD.get(p["tf"], 0.01),
                                  HORIZON_K.get(p["tf"], 1), source="manual")
        if p.get("session_id"):
            self.store.set_session_true_outcome(p["session_id"], label)
        self.store.mark_graded(p["id"], "manual")
        return {"status": "manual", "rewards": rewards, "realized_label": label}

    def _apply_correction(self, p: Dict[str, Any], label: str,
                          news_reward: Optional[float]) -> Dict[str, Any]:
        """Auto rewards were already applied; add (manual - auto) per child agent
        so the net policy effect equals the human verdict. (The brain's slow
        priority drift keeps the earlier auto signal — not worth un-winding.)"""
        prior_auto = {r["agent"]: r["reward"] for r in self.store.rewards_for(p["id"])
                      if r["source"] == "auto"}
        corrections: Dict[str, float] = {}

        if p.get("news_feats") is not None and p.get("news_action_idx") is not None:
            manual = news_reward if news_reward is not None else reward_for(p["news_action"], label)
            corr = manual - prior_auto.get("news", 0.0)
            if corr:
                self.dm.news.apply_reward(p["news_feats"], p["news_action_idx"], corr)
            self.store.record_reward(p["id"], "news", p["news_action"], corr, "correction")
            corrections["news"] = corr

        if p.get("research_feats") is not None and p.get("research_action_idx") is not None:
            manual = reward_for(p["research_action"], label)
            corr = manual - prior_auto.get("research", 0.0)
            if corr:
                self.dm.research.apply_reward(p["research_feats"], p["research_action_idx"], corr)
            self.store.record_reward(p["id"], "research", p["research_action"], corr, "correction")
            corrections["research"] = corr

        if p.get("indicator_blend") is not None:
            manual = reward_for(p["indicator_action"], label)
            corr = manual - prior_auto.get("indicator", 0.0)
            if corr:
                self.dm.indicator.apply_reward(p["indicator_blend"], corr)
            self.store.record_reward(p["id"], "indicator", p["indicator_action"], corr, "correction")
            corrections["indicator"] = corr

        if p.get("session_id"):
            self.store.set_session_true_outcome(p["session_id"], label)
        self.store.mark_graded(p["id"], "manual")
        return {"status": "corrected", "corrections": corrections, "realized_label": label}
