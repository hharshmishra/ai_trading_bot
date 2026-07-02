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

import config
from grading.barriers import triple_barrier
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


def _opposite(action: str) -> str:
    return "sell" if action == "buy" else "buy"


def reward_for_v2(predicted: Optional[str], realized: str) -> float:
    """Reward map v2 (TB_GRADING_ENABLED): separates direction errors from
    participation errors instead of the flat +1/-4.

      correct call (incl. skip when flat)      -> +REWARD_CORRECT   (+1.0)
      directional call, opposite realized      ->  REWARD_WRONG     (-4.0)
      directional call, market went nowhere    ->  REWARD_TIMEOUT_FLAT (-1.5)
      skip call, market moved decisively       ->  REWARD_MISSED_MOVE  (-1.0)
    """
    pred = _norm_action(predicted)
    if pred == realized:
        return config.REWARD_CORRECT
    if pred in ("buy", "sell") and realized == "skip":
        return config.REWARD_TIMEOUT_FLAT
    if pred == "skip" and realized in ("buy", "sell"):
        return config.REWARD_MISSED_MOVE
    return config.REWARD_WRONG


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
    def _path_after(self, pair: str, tf: str, candle_close_ts: Optional[float],
                    k: int) -> Optional[pd.DataFrame]:
        """OHLC candles STRICTLY after candle_close_ts (chronological), or None
        until at least k of them have printed. Shared by the fixed-horizon and
        triple-barrier graders so both see the same path."""
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
        after = df[ts > close_dt].reset_index(drop=True)
        if len(after) < k:
            return None
        return after

    def _realized_close(self, pair: str, tf: str, candle_close_ts: Optional[float], k: int) -> Optional[float]:
        """Close of the k-th candle strictly after candle_close_ts, or None if it
        has not printed yet."""
        after = self._path_after(pair, tf, candle_close_ts, k)
        if after is None:
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
        path = self._path_after(p["pair"], tf, p.get("candle_close_ts"), k)
        if path is None:
            return None  # horizon not reached yet; stays pending for a later pass
        fr = (float(path.iloc[k - 1]["close"]) - entry) / entry
        label_fixed = realized_label(fr, tf)

        # Triple-barrier labeling (Phase 3): rows recorded with barrier prices
        # get a path-aware label. Always RECORDED (shadow evidence); only drives
        # rewards when TB_GRADING_ENABLED. Legacy rows (NULL tp_price) keep the
        # fixed-horizon path forever.
        label_tb = hit_idx = exit_price = None
        final_action = _norm_action(p.get("final_action"))
        if p.get("tp_price") and p.get("sl_price") and final_action in ("buy", "sell"):
            out = triple_barrier(path, entry, final_action,
                                 float(p["tp_price"]), float(p["sl_price"]), k)
            if out.label_tb != "incomplete":
                label_tb, hit_idx, exit_price = out.label_tb, out.hit_idx, out.exit_price

        if config.TB_GRADING_ENABLED and label_tb in ("tp", "sl"):
            label = final_action if label_tb == "tp" else _opposite(final_action)
        else:
            label = label_fixed
        reward_fn = reward_for_v2 if config.TB_GRADING_ENABLED else reward_for

        rewards = self._apply_rewards(p, label, source="auto", reward_fn=reward_fn)
        self.store.record_outcome(p["id"], fr, label, th, k, source="auto",
                                  label_tb=label_tb, barrier_hit_idx=hit_idx,
                                  exit_price=exit_price)
        self.store.mark_graded(p["id"], "auto")
        return {"id": p["id"], "pair": p["pair"], "tf": tf, "forward_return": fr,
                "realized_label": label, "label_tb": label_tb, "rewards": rewards}

    def _apply_rewards(self, p: Dict[str, Any], label: str, source: str,
                       news_reward_override: Optional[float] = None,
                       reward_fn=reward_for) -> Dict[str, float]:
        """Apply per-agent rewards from the STORED payloads. News may take an
        explicit numeric reward (the Telegram 1.0/-4.0); the others match against
        the realized label via ``reward_fn`` (v1 flat map or v2 TB map)."""
        rewards: Dict[str, float] = {}

        if p.get("news_feats") is not None and p.get("news_action_idx") is not None:
            rn = news_reward_override if news_reward_override is not None else reward_fn(p["news_action"], label)
            self.dm.news.apply_reward(p["news_feats"], p["news_action_idx"], rn)
            self.store.record_reward(p["id"], "news", p["news_action"], rn, source)
            rewards["news"] = rn

        if p.get("research_feats") is not None and p.get("research_action_idx") is not None:
            rr = reward_fn(p["research_action"], label)
            self.dm.research.apply_reward(p["research_feats"], p["research_action_idx"], rr)
            self.store.record_reward(p["id"], "research", p["research_action"], rr, source)
            rewards["research"] = rr

        if p.get("indicator_blend") is not None:
            ri = reward_fn(p["indicator_action"], label)
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
