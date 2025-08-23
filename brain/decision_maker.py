#!/usr/bin/env python3
"""brain/decision_maker.py

Decision Maker (Brain) agent that orchestrates three child agents:
  - IndicatorAgent (high priority initially)
  - ResearchAgent
  - NewsAgent (lowest priority initially)

Features:
  - Calls child agents for a symbol+timeframe and aggregates their outputs
  - Maintains a simple persistent brain policy (scores -> normalized weights)
  - Prompts the user for ground-truth (buy/sell/skip) and a numeric reward for the news agent
  - Forwards feedback to child agents:
      * news_agent.learn(action_label, reward_float)
      * indicator_agent.learn(predicted_action, true_outcome_str)
      * research_agent.learn(predicted_action, true_outcome_str)
  - Updates internal agent scores and re-normalizes weights so the brain "learns" which agents to prioritize

This file is written to be dropped into your existing project under brain/decision_maker.py
It is intentionally dependency-light for orchestration; running decisions will call the agents which rely on your project's data fetching and indicator libraries.

Usage (simple):
    python decision_maker.py

You can also import DecisionMaker from other scripts.
"""

from __future__ import annotations
import json
import os
import time
from dataclasses import is_dataclass, asdict
from typing import Any, Dict, Optional, Tuple

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# import the child agents from your project
from agents.indicator_agent import IndicatorAgent
from agents.research_agent import ResearchAgent
from agents.news_agent import NewsAgent

LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
POLICY_PATH = os.path.join(LOG_DIR, "brain_policy.json")

# initial (relative) scores - indicator > research > news as you asked
DEFAULT_SCORES = {"indicator": 3.0, "research": 2.0, "news": 1.0}


def _ensure_logs_dir():
    os.makedirs(LOG_DIR, exist_ok=True)


def _load_policy() -> Dict[str, Any]:
    _ensure_logs_dir()
    if os.path.exists(POLICY_PATH):
        try:
            with open(POLICY_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    # default policy
    pol = {"scores": DEFAULT_SCORES.copy(), "weights": None, "updated_at": None}
    _save_policy(pol)
    return pol


def _save_policy(pol: Dict[str, Any]):
    pol["updated_at"] = time.time()
    with open(POLICY_PATH, "w", encoding="utf-8") as f:
        json.dump(pol, f, indent=2)


class DecisionMaker:
    def __init__(self, prefer_csv: bool = False):
        self.indicator = IndicatorAgent(prefer_csv=prefer_csv)
        self.research = ResearchAgent(prefer_csv=prefer_csv)
        self.news = NewsAgent()
        self.policy = _load_policy()
        self._normalize_weights()

    def _normalize_weights(self):
        # Convert raw scores -> normalized positive weights that sum to 1
        scores = self.policy.get("scores", DEFAULT_SCORES.copy())
        # ensure all keys exist
        for k in DEFAULT_SCORES.keys():
            scores.setdefault(k, DEFAULT_SCORES[k])
        # shift to positive
        minv = min(scores.values())
        shift = 0.0
        if minv <= 0:
            shift = abs(minv) + 0.01
        pos = {k: float(v) + shift for k, v in scores.items()}
        total = sum(pos.values()) or 1.0
        weights = {k: v / total for k, v in pos.items()}
        self.policy["scores"] = scores
        self.policy["weights"] = weights
        _save_policy(self.policy)

    @staticmethod
    def _normalize_action(a: Any) -> str:
        if a is None:
            return "skip"
        if isinstance(a, str):
            s = a.strip().lower()
            # Accept "BUY", "buy", "Buy" and also single-letter shortcuts
            if s in ("b", "buy", "bull", "bullish"):
                return "buy"
            if s in ("s", "sell", "bear", "bearish"):
                return "sell"
            if s in ("k", "skip", "hold", "none", "neutral"):
                return "skip"
            # numeric string?
            try:
                _ = float(s)
                return "skip"
            except Exception:
                return s
        # if an int (0,1,2) used by some agents:
        if isinstance(a, int):
            return {0: "sell", 1: "skip", 2: "buy"}.get(a, "skip")
        # fallback
        return str(a).lower()

    def _coerce_agent_out(self, raw_out: Any, agent_name: str) -> Dict[str, Any]:
        """Return a normalized dict: {action: 'buy'|'sell'|'skip', confidence: float, raw: raw_out}"""
        if raw_out is None:
            return {"action": "skip", "confidence": 0.0, "raw": None}

        # IndicatorAgent returns a dataclass (IndicatorDecision) or dict-like
        if agent_name == "indicator":
            if is_dataclass(raw_out):
                dd = asdict(raw_out)
            elif isinstance(raw_out, dict):
                dd = raw_out
            else:
                dd = getattr(raw_out, "__dict__", dict(raw_out))
            action = self._normalize_action(dd.get("action"))
            confidence = float(dd.get("confidence", 0.0) or 0.0)
            return {"action": action, "confidence": confidence, "raw": dd}

        # ResearchAgent returns a dict
        if agent_name == "research":
            dd = raw_out if isinstance(raw_out, dict) else getattr(raw_out, "__dict__", {"action": None, "confidence": 0.0})
            action = self._normalize_action(dd.get("action"))
            confidence = float(dd.get("confidence", 0.0) or 0.0)
            return {"action": action, "confidence": confidence, "raw": dd}

        # NewsAgent returns a dict: {"action": "BUY"/"SELL"/"SKIP", "confidence": float, ...}
        if agent_name == "news":
            dd = raw_out if isinstance(raw_out, dict) else getattr(raw_out, "__dict__", {"action": None, "confidence": 0.0})
            action = self._normalize_action(dd.get("action"))
            confidence = float(dd.get("confidence", 0.0) or 0.0)
            return {"action": action, "confidence": confidence, "raw": dd}

        # fallback
        return {"action": "skip", "confidence": 0.0, "raw": raw_out}

    def decide(self, symbol: str, timeframe: str, use_agents: Optional[Tuple[str, ...]] = ("indicator", "research", "news")) -> Dict[str, Any]:
        """Call child agents according to use_agents and aggregate a final decision."""
        agent_results: Dict[str, Dict[str, Any]] = {}

        # call indicator agent
        ind_out = None
        if "indicator" in use_agents:
            try:
                ind_out = self.indicator.decide(symbol, timeframe)
            except Exception as e:
                ind_out = None
        agent_results["indicator"] = self._coerce_agent_out(ind_out, "indicator")

        # call research agent (it can accept references to other agents)
        res_out = None
        if "research" in use_agents:
            try:
                res_out = self.research.decide(symbol, timeframe, indicator_agent=self.indicator, news_agent=self.news)
            except Exception as e:
                res_out = None
        agent_results["research"] = self._coerce_agent_out(res_out, "research")

        # call news agent
        news_out = None
        if "news" in use_agents:
            try:
                news_out = self.news.run(symbol)
            except Exception as e:
                news_out = None
        agent_results["news"] = self._coerce_agent_out(news_out, "news")

        # Weighted aggregation
        weights = self.policy.get("weights", {"indicator": 0.6, "research": 0.3, "news": 0.1})
        action_map = {"sell": -1.0, "skip": 0.0, "buy": 1.0}

        # compute score = sum(weight * action_value * confidence)
        total_score = 0.0
        total_weighted_conf = 0.0
        for ag in ("indicator", "research", "news"):
            ag_res = agent_results.get(ag, {"action": "skip", "confidence": 0.0})
            val = action_map.get(ag_res["action"], 0.0)
            w = float(weights.get(ag, 0.0))
            conf = float(ag_res.get("confidence", 0.0) or 0.0)
            total_score += w * val * conf
            total_weighted_conf += w * conf

        final_confidence = float(abs(total_score) / total_weighted_conf) if total_weighted_conf > 0 else 0.0
        # decision thresholds: small deadzone -> skip
        if total_score > 0.05:
            final_action = "buy"
        elif total_score < -0.05:
            final_action = "sell"
        else:
            final_action = "skip"

        result = {
            "chartName": symbol,
            "timeframe": timeframe,
            "agents": agent_results,
            "final": {"action": final_action, "confidence": round(final_confidence, 4), "score": round(total_score, 6)},
            "policy": {"scores": self.policy.get("scores"), "weights": self.policy.get("weights")},
        }
        return result
    
    #REACTIVE TO EVERY AWARD/PUNISHMENT

    # def _apply_feedback_to_brain(self, agent_results: Dict[str, Dict[str, Any]], true_outcome: str, news_reward: float):
    #     """Update internal brain scores using child's correctness and confidence, persist weights."""
    #     true = self._normalize_action(true_outcome)
    #     # update scores: +1 for correct, -4 for wrong, scaled by confidence
    #     for ag in ("indicator", "research", "news"):
    #         res = agent_results.get(ag) or {"action": "skip", "confidence": 0.0}
    #         pred = res.get("action", "skip")
    #         conf = float(res.get("confidence", 0.0) or 0.0)
    #         # reward for brain: use same scheme as agents (1 / -4) scaled by confidence
    #         delta = (1.0 if pred == true else -4.0) * conf
    #         self.policy["scores"][ag] = float(self.policy["scores"].get(ag, 0.0)) + delta

    #     # additionally incorporate the explicit numeric news_reward directly into news score
    #     try:
    #         self.policy["scores"]["news"] = float(self.policy["scores"].get("news", 0.0)) + float(news_reward)
    #     except Exception:
    #         pass

    #     # re-normalize weights and save
    #     self._normalize_weights() 
    
    def _apply_feedback_to_brain(self, agent_results: Dict[str, Dict[str, Any]], true_outcome: str, news_reward: float):
        """Update scores slowly so Indicator > Research > News stays stable unless
        long-term evidence suggests otherwise."""
        true = self._normalize_action(true_outcome)

        # learning rate controls how fast priorities can change
        LEARNING_RATE = 0.05  # 5% adjustment per feedback

        for ag in ("indicator", "research", "news"):
            res = agent_results.get(ag) or {"action": "skip", "confidence": 0.0}
            pred = res.get("action", "skip")
            conf = float(res.get("confidence", 0.0) or 0.0)

            delta = (1.0 if pred == true else -4.0) * conf
            # apply with slow drift
            self.policy["scores"][ag] = (
                float(self.policy["scores"].get(ag, 0.0))
                + LEARNING_RATE * delta
            )

        # news agent also gets explicit reward (scaled)
        try:
            self.policy["scores"]["news"] += LEARNING_RATE * float(news_reward)
        except Exception:
            pass

        self._normalize_weights()


    def feedback(self, decision_out: Dict[str, Any]):
        """Interactive prompt: ask user for true outcome and numeric news reward and forward to child agents."""
        agents = decision_out["agents"]
        # show summary
        print("\n=== Decision summary ===")
        print(json.dumps(decision_out, indent=2))
        print("========================\n")

        # ask for true outcome (buy/sell/skip)
        true = input("Enter true outcome (buy/sell/skip) for this chart (or blank to skip learning): ").strip()
        if true == "":
            print("Skipping learning for this chart.")
            return

        true = self._normalize_action(true)
        # ask for numeric reward for news agent
        nr_in = input("Enter numeric reward for news agent (e.g. 1.0 or -4.0). Press Enter to auto-assign based on match: ").strip()
        if nr_in == "":
            # auto assign: if news matched true => +1 else -4
            news_pred = agents.get("news", {}).get("action")
            news_reward = 1.0 if news_pred == true else -4.0
            print(f"Auto news reward => {news_reward}")
        else:
            try:
                news_reward = float(nr_in)
            except Exception:
                print("Invalid numeric reward, falling back to -4.0")
                news_reward = -4.0

        # Forward feedback to child agents
        # NewsAgent: learn(action_label, reward_float)
        try:
            news_raw = agents.get("news", {}).get("raw", {})
            # news_agent.learn expects the textual label from last run (e.g., 'BUY'/'SELL'/'SKIP') or None
            news_action_label = None
            if isinstance(news_raw, dict) and "action" in news_raw:
                news_action_label = news_raw.get("action")
            self.news.learn(news_action_label, reward=news_reward)
        except Exception as e:
            print("Warning: news_agent.learn failed:", e)

        # IndicatorAgent: learn(predicted_action, true_outcome)
        try:
            ind_pred = agents.get("indicator", {}).get("action")
            self.indicator.learn(predicted_action=ind_pred, true_outcome=true)
        except Exception as e:
            print("Warning: indicator_agent.learn failed:", e)

        # ResearchAgent: learn(predicted_action, true_outcome) - research.learn can accept either true_outcome or reward
        try:
            res_pred = agents.get("research", {}).get("action")
            self.research.learn(predicted_action=res_pred, true_outcome=true, reward=None)
        except Exception:
            # some versions accept (pred, true_outcome) directly under other param names
            try:
                self.research.learn(res_pred, true)
            except Exception as e:
                print("Warning: research_agent.learn failed:", e)

        # Update brain policy scores & weights
        try:
            self._apply_feedback_to_brain(agents, true, news_reward)
            print("Brain policy updated. New weights:", json.dumps(self.policy.get("weights", {}), indent=2))
        except Exception as e:
            print("Warning: failed to update brain policy:", e)


def demo_run():
    dm = DecisionMaker(prefer_csv=False)
    # default symbols and timeframes - you can change this or wire it to your main.py
    symbols = ["LINKUSDT", "POLUSDT"]
    timeframes = ["4h"]

    for s in symbols:
        for tf in timeframes:
            print(f"\n--- Running brain for {s} @ {tf} ---")
            out = dm.decide(s, tf)
            dm.feedback(out)


if __name__ == "__main__":
    demo_run()
