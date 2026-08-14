#!/usr/bin/env python3
"""brain/decision_maker.py

Decision Maker (Brain): orchestrates the five voter agents — Indicator,
Research, News, Derivatives, Sentiment — for a symbol+timeframe and
aggregates their votes with learned priority weights (persistent policy:
scores -> normalized weights).

Learning: the grader calls ``apply_brain_feedback(agent_results, label)``
after every graded prediction; agent scores drift toward the voters that
were right and the weights renormalize. Per-agent policy updates happen in
each agent's own stateless ``apply_reward`` — never here.
"""

from __future__ import annotations
import json
import math
import os
import time
from dataclasses import is_dataclass, asdict
from typing import Any, Dict, Optional, Tuple

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# import the child agents from your project
import config
from agents.indicator_agent import IndicatorAgent
from agents.research_agent import ResearchAgent
from agents.news_agent import NewsAgent

LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
POLICY_PATH = os.path.join(LOG_DIR, "brain_policy.json")

# Voter roster. Every aggregation / feedback loop iterates this tuple, so adding
# a voter is one entry here + a DEFAULT_SCORES prior.
AGENT_NAMES = ("indicator", "research", "news", "derivatives", "sentiment")

# initial (relative) scores - indicator > research > news as you asked;
# derivatives/sentiment start between research and news (informative, unproven).
DEFAULT_SCORES = {"indicator": 3.0, "research": 2.0, "news": 1.0,
                  "derivatives": config.DERIV_BRAIN_SCORE,
                  "sentiment": config.SENTIMENT_BRAIN_SCORE}

# Trust dynamics (v3.7). Scores are clamped so no voter is ever
# mathematically unrecoverable (prod v3.6 let indicator drift to -1525 —
# ~30k net-correct rows to climb back), and weights come from a softmax so
# they rank agents absolutely, not by distance from the worst one (the old
# shift-normalize crowned a never-voting agent with top trust).
TRUST_TEMPERATURE = 2.0   # softmax temperature over trust scores
SCORE_CLAMP = 10.0        # trust scores live in [-10, +10]
WEIGHT_FLOOR = 0.02       # every voter keeps >= 2% voice
TRUST_LR = 0.05           # 5% adjustment per feedback
# v3.8: EMA horizon of the per-agent advantage baseline (~50 directional
# votes). The v3.7 symmetric map was still negative-sum at realized base
# rates (~28% correct / 29% opposite / 43% flat => E ~ -0.11*conf per vote):
# in 21 prod days every score sank and derivatives pinned at the -10 rail.
# Subtracting each agent's own running average makes trust zero-sum around
# "your recent self" — only being BETTER than your own base rate raises trust.
TRUST_BASELINE_ALPHA = 0.02


def brain_trust_outcome(pred: str, realized: str) -> Optional[float]:
    """Direction-quality outcome score for one vote, or None for skip votes
    (they carry no trust evidence — they already hold no vote mass)."""
    if pred not in ("buy", "sell"):
        return None
    if realized == pred:
        return 1.0
    if realized == "skip":
        return -0.25
    return -1.0


def brain_trust_delta(pred: str, realized: str, conf: float,
                      baseline: float = 0.0) -> float:
    """Advantage-style trust signal, decoupled from the bandit reward map.

    The bandits keep the asymmetric v2 map (-4 confidently-wrong etc.); trust
    cannot, because at realized base rates that map bankrupts EVERY directional
    voter and the brain converges on whoever abstains. Trust asks one question:
    when this agent speaks, is the direction right MORE OFTEN THAN ITS OWN
    RECENT AVERAGE? delta = conf * (outcome_score - baseline), outcome_score
    = +1 correct / -1 opposite / -0.25 flat; skip votes -> 0.
    """
    out = brain_trust_outcome(pred, realized)
    if out is None:
        return 0.0
    return conf * (out - baseline)


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
    def __init__(self, prefer_csv: bool = False, store=None):
        self.indicator = IndicatorAgent(prefer_csv=prefer_csv)
        self.research = ResearchAgent(prefer_csv=prefer_csv)
        self.news = NewsAgent()
        # RAG store for headline grounding (correctness v3, A4). Lazy default:
        # resolved on first use so tests with mocked agents pay nothing.
        self._store = store
        # 4th voter (Phase 4): no-LLM positioning agent; brain calls it only
        # when DERIVATIVES_ENABLED, but the attribute always exists so the
        # grader can replay rewards on rows that carry deriv snapshots.
        from agents.derivatives_agent import DerivativesAgent
        self.derivatives = DerivativesAgent(data_fetcher=self.indicator.data)
        # 5th voter (v3.5): crowd sentiment + on-chain + taker flow; same
        # always-instantiated rule so the grader can replay stored snapshots.
        from agents.sentiment_agent import SentimentAgent
        self.sentiment = SentimentAgent(data_fetcher=self.indicator.data)
        self.policy = _load_policy()
        self._normalize_weights()

    def _normalize_weights(self):
        # Raw scores -> softmax(score / T) weights with a per-agent floor.
        scores = self.policy.get("scores", DEFAULT_SCORES.copy())
        # ensure all keys exist (setdefault absorbs newly added voters like
        # "derivatives" into a pre-existing policy file — no migration needed)
        for k in DEFAULT_SCORES.keys():
            scores.setdefault(k, DEFAULT_SCORES[k])
        # clamp persists back into the policy: self-heals legacy files whose
        # scores drifted unboundedly under the pre-v3.7 dynamics
        scores = {k: max(-SCORE_CLAMP, min(SCORE_CLAMP, float(v)))
                  for k, v in scores.items()}
        exps = {k: math.exp(v / TRUST_TEMPERATURE) for k, v in scores.items()}
        total = sum(exps.values()) or 1.0
        weights = {k: v / total for k, v in exps.items()}
        # exact floor: floored agents get WEIGHT_FLOOR, the rest re-scale into
        # the remaining mass; loop because re-scaling can push another under
        floored: set = set()
        while len(floored) < len(weights):
            rest = sum(w for k, w in weights.items() if k not in floored)
            room = 1.0 - WEIGHT_FLOOR * len(floored)
            scale = room / rest if rest > 0 else 0.0
            newly = {k for k in weights
                     if k not in floored and weights[k] * scale < WEIGHT_FLOOR}
            if not newly:
                weights = {k: WEIGHT_FLOOR if k in floored else weights[k] * scale
                           for k in weights}
                break
            floored |= newly
        else:
            weights = {k: 1.0 / len(weights) for k in weights}
        self.policy["scores"] = scores
        self.policy["weights"] = weights
        _save_policy(self.policy)

    def _headlines_for(self, symbol: str) -> Optional[list]:
        """Fresh stored headline titles for a symbol's base asset (48h, top 5),
        or None — never raises, never blocks a decision."""
        if not config.NEWS_RAG_ENABLED:
            return None
        try:
            if self._store is None:
                from persistence import get_store
                self._store = get_store()
            from agents.research_agent import _strip_suffix
            from ingestion import format_headline
            rows = self._store.recent_news_for_asset(
                _strip_suffix(symbol), since_ts=time.time() - 48 * 3600, limit=5)
            titles = [format_headline(r) for r in rows if r.get("title")]
            return titles or None
        except Exception:
            return None

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

        # Derivatives/Sentiment agents return dicts; unavailable -> conf 0.0
        if agent_name in ("derivatives", "sentiment"):
            dd = raw_out if isinstance(raw_out, dict) else getattr(raw_out, "__dict__", {"action": None, "confidence": 0.0})
            action = self._normalize_action(dd.get("action"))
            confidence = float(dd.get("confidence", 0.0) or 0.0)
            return {"action": action, "confidence": confidence, "raw": dd}

        # fallback
        return {"action": "skip", "confidence": 0.0, "raw": raw_out}

    def decide(self, symbol: str, timeframe: str, use_agents: Optional[Tuple[str, ...]] = None, market_context: Optional[Any] = None) -> Dict[str, Any]:
        """Call child agents according to use_agents and aggregate a final decision.

        ``market_context`` (a shared MarketContext built once per cycle) is passed
        to the research agent and reused for the news overall scan, so a full
        cycle costs ~73 LLM calls instead of ~576. Without it, the original
        per-coin path runs unchanged.

        ``use_agents`` defaults to the full roster; the derivatives voter also
        requires DERIVATIVES_ENABLED (flag off => identical 3-voter behaviour).
        """
        if use_agents is None:
            use_agents = AGENT_NAMES
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
                res_out = self.research.decide(symbol, timeframe, indicator_agent=self.indicator, news_agent=self.news, market_context=market_context)
            except Exception as e:
                res_out = None
        agent_results["research"] = self._coerce_agent_out(res_out, "research")

        # call news agent (reuse the shared overall scan when a context is given;
        # ground the pair scan in stored headlines — correctness v3, A4)
        news_out = None
        if "news" in use_agents:
            try:
                shared_overall = market_context.overall_json if market_context is not None else None
                news_out = self.news.run(symbol, overall_json=shared_overall,
                                         headlines=self._headlines_for(symbol))
            except Exception as e:
                news_out = None
        agent_results["news"] = self._coerce_agent_out(news_out, "news")

        # call derivatives agent (Phase 4; flag-gated, keyless public data)
        deriv_out = None
        if "derivatives" in use_agents and config.DERIVATIVES_ENABLED:
            try:
                deriv_out = self.derivatives.decide(symbol, timeframe)
            except Exception:
                deriv_out = None
        agent_results["derivatives"] = self._coerce_agent_out(deriv_out, "derivatives")

        # call sentiment agent (v3.5; flag-gated, keyless free data)
        sent_out = None
        if "sentiment" in use_agents and config.SENTIMENT_ENABLED:
            try:
                sent_out = self.sentiment.decide(symbol, timeframe,
                                                 market_context=market_context)
            except Exception:
                sent_out = None
        agent_results["sentiment"] = self._coerce_agent_out(sent_out, "sentiment")

        # Weighted aggregation over the ACTIVE roster only — a voter disabled
        # by flag (or excluded via use_agents) must not hold vote mass, or its
        # dead weight shrinks everyone else's say against the ±0.05 deadzone.
        # final_confidence is invariant to this renormalization (ratio).
        weights = self.policy.get("weights", {"indicator": 0.6, "research": 0.3, "news": 0.1})
        active = [ag for ag in AGENT_NAMES if ag in use_agents]
        if not config.DERIVATIVES_ENABLED and "derivatives" in active:
            active.remove("derivatives")
        if not config.SENTIMENT_ENABLED and "sentiment" in active:
            active.remove("sentiment")
        wsum = sum(float(weights.get(ag, 0.0)) for ag in active) or 1.0
        action_map = {"sell": -1.0, "skip": 0.0, "buy": 1.0}

        # compute score = sum(weight * action_value * confidence)
        total_score = 0.0
        total_weighted_conf = 0.0
        for ag in active:
            ag_res = agent_results.get(ag, {"action": "skip", "confidence": 0.0})
            val = action_map.get(ag_res["action"], 0.0)
            w = float(weights.get(ag, 0.0)) / wsum
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

        # Deadzone v2 (correctness v3, A7): a single weak voter can push
        # |score| past 0.05 while three high-confidence skips disagree —
        # v2 additionally requires a confidence floor. Always computed and
        # stamped (shadow column final_action_v2); only DRIVES the final
        # action when BRAIN_DEADZONE_V2 is on, after live shadow evidence.
        action_v2 = final_action
        if final_action != "skip" and final_confidence < config.BRAIN_MIN_CONF:
            action_v2 = "skip"
        if config.BRAIN_DEADZONE_V2:
            final_action = action_v2

        result = {
            "chartName": symbol,
            "timeframe": timeframe,
            "agents": agent_results,
            "final": {"action": final_action, "confidence": round(final_confidence, 4),
                      "score": round(total_score, 6), "action_v2": action_v2},
            "policy": {"scores": self.policy.get("scores"), "weights": self.policy.get("weights")},
        }
        return result
    
    def _apply_feedback_to_brain(self, agent_results: Dict[str, Dict[str, Any]], true_outcome: str):
        """Update scores slowly so Indicator > Research > News stays stable
        unless long-term evidence suggests otherwise. Trust uses the symmetric
        direction-quality map (``brain_trust_delta``) — NOT the bandit reward
        map — and scores stay clamped so recovery is always possible. Each
        agent is scored exactly once.
        """
        true = self._normalize_action(true_outcome)
        baselines = self.policy.setdefault("baseline", {})
        for ag in AGENT_NAMES:
            res = agent_results.get(ag)
            if res is None:
                continue  # e.g. pre-Phase-4 rows without a derivatives snapshot
            pred = self._normalize_action(res.get("action", "skip"))
            conf = float(res.get("confidence", 0.0) or 0.0)
            out = brain_trust_outcome(pred, true)
            if out is None:
                continue  # skip votes carry no trust evidence
            b = float(baselines.get(ag, 0.0))
            delta = conf * (out - b)
            # baseline updates AFTER the delta so the first vote after a reset
            # is judged against 0 (neutral), not against itself
            baselines[ag] = (1.0 - TRUST_BASELINE_ALPHA) * b + TRUST_BASELINE_ALPHA * out
            s = float(self.policy["scores"].get(ag, 0.0)) + TRUST_LR * delta
            self.policy["scores"][ag] = max(-SCORE_CLAMP, min(SCORE_CLAMP, s))

        self._normalize_weights()

    def apply_brain_feedback(self, agent_results: Dict[str, Dict[str, Any]], true_outcome: str):
        """Public entry for the grader / Telegram handler to update the brain's
        agent-priority weights from a (auto- or human-derived) ground truth.
        ``agent_results`` = {agent: {"action": str, "confidence": float}}.
        """
        self._apply_feedback_to_brain(agent_results, true_outcome)

    def decay_trust(self, factor: Optional[float] = None):
        """Nightly geometric pull of every trust score toward 0 (v3.8).

        The +/-10 clamp made scores recoverable but not un-stuck: a voter
        pinned at a rail needs the same volume of counter-evidence back.
        Decay guarantees rails are temporary (0.98/night ~ half-life 34 days)
        while leaving day-to-day ordering intact. Factor <=0 or >=1 no-ops.
        """
        f = config.TRUST_DECAY if factor is None else float(factor)
        if not (0.0 < f < 1.0):
            return
        scores = self.policy.get("scores") or {}
        self.policy["scores"] = {k: float(v) * f for k, v in scores.items()}
        self._normalize_weights()
