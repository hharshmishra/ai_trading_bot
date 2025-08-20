import json
import os
import random
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, List, Tuple

from dotenv import load_dotenv
load_dotenv()

# --- OpenAI (official SDK) ---
# pip install openai>=1.30.0
from openai import OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- LangChain / LangGraph ---
# pip install langchain langgraph pydantic
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field

POLICY_PATH = "logs/news_agent_policy.json"
os.makedirs("logs", exist_ok=True)


# =========================
# JSON SCHEMAS (Pydantic)
# =========================

class PanicHeadline(BaseModel):
    title: str = Field(..., description="Headline title")
    impact: str = Field(..., description="Impact direction: Bullish, Bearish, or Neutral")
    reason: str = Field(..., description="Why it could move markets")

class OverallScanJSON(BaseModel):
    has_panic: bool = Field(..., description="True if any panic-worthy headlines found")
    sentiment: str = Field(..., description="Overall sentiment: Bullish/Bearish/Neutral")
    confidence: float = Field(..., ge=0, le=1, description="Confidence for overall sentiment 0..1")
    top_headlines: List[PanicHeadline] = Field(default_factory=list)

class PairHeadline(BaseModel):
    title: str
    impact: str
    reason: str

class PairScanJSON(BaseModel):
    pair: str = Field(..., description="Trading pair analysed, e.g., BTCUSDT")
    sentiment: str = Field(..., description="Pair-specific sentiment")
    confidence: float = Field(..., ge=0, le=1)
    top_headlines: List[PairHeadline] = Field(default_factory=list)


# =========================
# RL: Tiny Contextual Bandit
# =========================
@dataclass
class BanditPolicy:
    # Linear model: action logits = W · features
    # actions = [SELL=0, SKIP=1, BUY=2]
    weights: List[List[float]]  # 3 x F
    epsilon: float  # exploration rate

    @staticmethod
    def default(n_features: int) -> "BanditPolicy":
        # Small random init for stability
        rng = random.Random(42)
        weights = [[rng.uniform(-0.05, 0.05) for _ in range(n_features)] for _ in range(3)]
        return BanditPolicy(weights=weights, epsilon=0.1)


def dot(a: List[float], b: List[float]) -> float:
    return sum(x*y for x, y in zip(a, b))


def softmax(logits: List[float]) -> List[float]:
    import math
    m = max(logits)
    exps = [math.exp(l - m) for l in logits]
    s = sum(exps)
    return [e/s for e in exps]


class NewsRL:
    """
    Minimal contextual bandit:
      - Features: [overall_score, pair_score, panic_flag, bias_overall, bias_pair]
      - Actions: 0=SELL, 1=SKIP, 2=BUY
      - Policy: linear logits + softmax, epsilon-greedy pick
      - Update: REINFORCE-like gradient step proportional to reward
    """
    def __init__(self, n_features: int = 5, lr: float = 0.1):
        self.n_features = n_features
        self.lr = lr
        self.policy = self._load_policy()

    def _load_policy(self) -> BanditPolicy:
        if os.path.exists(POLICY_PATH):
            try:
                with open(POLICY_PATH, "r") as f:
                    data = json.load(f)
                return BanditPolicy(weights=data["weights"], epsilon=data.get("epsilon", 0.1))
            except Exception:
                pass
        return BanditPolicy.default(self.n_features)

    def _save_policy(self):
        with open(POLICY_PATH, "w") as f:
            json.dump({"weights": self.policy.weights, "epsilon": self.policy.epsilon}, f)

    def _logits(self, features: List[float]) -> List[float]:
        return [dot(w, features) for w in self.policy.weights]

    def select_action(self, features: List[float]) -> int:
        # epsilon-greedy
        if random.random() < self.policy.epsilon:
            return random.choice([0, 1, 2])
        logits = self._logits(features)
        probs = softmax(logits)
        # sample from probs
        r = random.random()
        c = 0.0
        for i, p in enumerate(probs):
            c += p
            if r <= c:
                return i
        return 2  # fallback BUY

    def update(self, features: List[float], action: int, reward: float):
        # Policy gradient step (simple)
        logits = self._logits(features)
        probs = softmax(logits)
        # gradient for chosen action = (1 - p_a)*x, for others = (-p_i)*x
        for a in range(3):
            grad_coeff = (1.0 if a == action else 0.0) - probs[a]
            for j in range(self.n_features):
                self.policy.weights[a][j] += self.lr * reward * grad_coeff * features[j]
        self._save_policy()


# =========================
# Prompt Templates
# =========================

OVERALL_PROMPT = PromptTemplate.from_template(
    """
You are a crypto news risk scanner. Analyze **current** crypto/market headlines and return JSON.

Goal:
- Find up to 5 **panic-worthy** headlines that could strongly move the market (positive or negative).
- If none exist, mark has_panic=false and be neutral.

Return a strict JSON object with fields:
{{
  "has_panic": boolean,
  "sentiment": "Bullish" | "Bearish" | "Neutral",
  "confidence": number between 0 and 1,
  "top_headlines": [
    {{ "title": str, "impact": "Bullish"|"Bearish"|"Neutral", "reason": str }},
    ...
  ]
}}

Only return JSON. No extra text.
"""
)

PAIR_PROMPT = PromptTemplate.from_template(
    """
You are a crypto pair-focused news analyzer.
Given a trading pair: "{pair}", analyze the **latest** top 8 headlines most relevant to this pair.
Decide pair-specific sentiment.

Return a strict JSON object:
{{
  "pair": "{pair}",
  "sentiment": "Bullish" | "Bearish" | "Neutral",
  "confidence": number between 0 and 1,
  "top_headlines": [
    {{ "title": str, "impact": "Bullish"|"Bearish"|"Neutral", "reason": str }},
    ...
  ]
}}

Only return JSON. No extra text.
"""
)

def _chat_json(prompt: str) -> Dict[str, Any]:
    """
    Call OpenAI with JSON-only output. Requires OPENAI_API_KEY.
    Adjust model name if needed.
    """
    resp = client.chat.completions.create(
        model="gpt-4o-mini",  # use any JSON-capable model you have access to
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    content = resp.choices[0].message.content
    return json.loads(content)


# =========================
# LangGraph: two-node graph
# =========================

class NewsGraphState(BaseModel):
    pair: str
    overall_json: Dict[str, Any] | None = None
    pair_json: Dict[str, Any] | None = None


def overall_scan_node(state: NewsGraphState) -> NewsGraphState:
    prompt = OVERALL_PROMPT.format()
    data = _chat_json(prompt)
    # validate
    OverallScanJSON.model_validate(data)
    state.overall_json = data
    return state

def pair_scan_node(state: NewsGraphState) -> NewsGraphState:
    prompt = PAIR_PROMPT.format(pair=state.pair)
    data = _chat_json(prompt)
    PairScanJSON.model_validate(data)
    state.pair_json = data
    return state


graph_builder = StateGraph(NewsGraphState)
graph_builder.add_node("overall_scan", RunnableLambda(overall_scan_node))
graph_builder.add_node("pair_scan", RunnableLambda(pair_scan_node))
graph_builder.set_entry_point("overall_scan")
graph_builder.add_edge("overall_scan", "pair_scan")
graph_builder.add_edge("pair_scan", END)
NEWS_GRAPH = graph_builder.compile()


# =========================
# Utility: scoring + action
# =========================

IMPACT_MAP = {"Bullish": 1, "Bearish": -1, "Neutral": 0}

def score_from_sentiment(sentiment: str, confidence: float) -> float:
    return IMPACT_MAP.get(sentiment, 0) * float(confidence)


def features_from_jsons(overall: OverallScanJSON, pairj: PairScanJSON) -> List[float]:
    overall_score = score_from_sentiment(overall.sentiment, overall.confidence)
    pair_score = score_from_sentiment(pairj.sentiment, pairj.confidence)
    panic_flag = 1.0 if overall.has_panic else 0.0
    bias_overall = IMPACT_MAP.get(overall.sentiment, 0)
    bias_pair = IMPACT_MAP.get(pairj.sentiment, 0)
    return [overall_score, pair_score, panic_flag, float(bias_overall), float(bias_pair)]


def action_to_label(action: int) -> str:
    # 0=SELL, 1=SKIP, 2=BUY
    return ["SELL", "SKIP", "BUY"][action]


# =========================
# Public Agent API
# =========================

class NewsAgent:
    """
    Two-stage GPT news agent with LangGraph + tiny RL.
    Usage:
        agent = NewsAgent()
        result = agent.run(pair="BTCUSDT")
        # later when ground-truth known:
        agent.learn(result["action"], reward=+1 or -4)
    """
    def __init__(self):
        self._rl = NewsRL(n_features=5)
        self._last_features: List[float] | None = None
        self._last_action: int | None = None
        self._last_pair: str | None = None
        self._last_raw: Dict[str, Any] | None = None

    def run(self, pair: str) -> Dict[str, Any]:
        state = NEWS_GRAPH.invoke({"pair": pair})
        overall = OverallScanJSON.model_validate(state["overall_json"])
        pairj = PairScanJSON.model_validate(state["pair_json"])

        feats = features_from_jsons(overall, pairj)
        action = self._rl.select_action(feats)

        self._last_features = feats
        self._last_action = action
        self._last_pair = pair
        self._last_raw = {"overall": overall.model_dump(), "pair": pairj.model_dump()}

        result = {
            "agent": "news",
            "pair": pair,
            "overall_json": overall.model_dump(),
            "pair_json": pairj.model_dump(),
            "action": action_to_label(action),
            "confidence": max(overall.confidence, pairj.confidence),
            "timestamp": datetime.utcnow().isoformat()
        }

        # Also write a line-log for traceability
        with open("logs/predictions_log.json", "a") as f:
            f.write(json.dumps({"type": "news_agent", **result}) + "\n")

        return result

    def learn(self, taken_action_label: str | None, reward: float):
        """
        Call this AFTER you know if the news decision helped or hurt.
        If you pass taken_action_label, it will map to action index; if None, uses last action.
        Reward: your scheme (+1 for correct, -4 for wrong) as you defined.
        """
        if self._last_features is None:
            return

        if taken_action_label is not None:
            label_map = {"SELL": 0, "SKIP": 1, "BUY": 2}
            action_idx = label_map.get(taken_action_label.upper(), self._last_action)
        else:
            action_idx = self._last_action

        if action_idx is None:
            return

        self._rl.update(self._last_features, action_idx, float(reward))

        # Optional: log the learning step
        learn_log = {
            "type": "news_agent_learn",
            "pair": self._last_pair,
            "features": self._last_features,
            "action": action_to_label(action_idx),
            "reward": reward,
            "timestamp": datetime.utcnow().isoformat()
        }
        with open("logs/predictions_log.json", "a") as f:
            f.write(json.dumps(learn_log) + "\n")


# =========================
# Quick local test (optional)
# =========================
if __name__ == "__main__":
    agent = NewsAgent()
    out = agent.run("BTCUSDT")
    print(json.dumps(out, indent=2))
    user_input_str = input("Enter a floating-point number: ")
    float_number = float(user_input_str)
    # later, after you verify outcome, call:
    agent.learn(out["action"], reward=float_number)   # or -4.0