"""DerivativesAgent — 4th brain voter (accuracy upgrade Phase 4). No LLM.

Reads Binance USDM positioning (funding rate, open interest, top-trader
long/short ratios) and votes buy/sell/skip through the same 3-action linear
softmax bandit the ResearchAgent uses. The thesis it can learn: crowded
positioning at funding extremes precedes squeezes — e.g. heavily positive
funding + rising OI = crowded longs = downside squeeze fuel.

Pairs without a USDM future (or any fetch error) return confidence 0.0 with
``available: False`` — a mathematical no-op in the brain's weighted sum, and
NULL feats mean the grader applies no reward.
"""
from __future__ import annotations

import json
import math
import os
import random
from typing import Any, Dict, List, Optional

import config
from utils import derivatives_fetcher as dfx

POLICY_PATH = "logs/derivatives_agent_policy.json"

N_FEATURES = 8
_ACTIONS = ["sell", "skip", "buy"]


def _softmax(logits: List[float]) -> List[float]:
    m = max(logits)
    exps = [math.exp(v - m) for v in logits]
    s = sum(exps)
    return [e / s for e in exps]


def _dot(a: List[float], b: List[float]) -> float:
    return float(sum(x * y for x, y in zip(a, b)))


class DerivativesRL:
    """3 x N_FEATURES linear softmax bandit (mirrors ResearchRL)."""

    def __init__(self, n_features: int = N_FEATURES, lr: float = 0.05,
                 policy_path: str = POLICY_PATH):
        self.n_features = n_features
        self.lr = lr
        self.policy_path = policy_path
        self.epsilon = 0.08
        self.weights = self._load()

    def _default_weights(self) -> List[List[float]]:
        rng = random.Random(42)
        return [[rng.uniform(-0.05, 0.05) for _ in range(self.n_features)]
                for _ in range(3)]

    def _load(self) -> List[List[float]]:
        try:
            if os.path.exists(self.policy_path) and os.path.getsize(self.policy_path) > 0:
                with open(self.policy_path, "r", encoding="utf-8") as f:
                    p = json.load(f)
                self.epsilon = float(p.get("epsilon", 0.08))
                w = p["weights"]
                if len(w) == 3 and len(w[0]) == self.n_features:
                    return w
        except Exception:
            pass
        w = self._default_weights()
        self._save(w)
        return w

    def _save(self, weights: Optional[List[List[float]]] = None) -> None:
        os.makedirs(os.path.dirname(self.policy_path) or ".", exist_ok=True)
        with open(self.policy_path, "w", encoding="utf-8") as f:
            json.dump({"weights": weights or self.weights, "epsilon": self.epsilon}, f, indent=2)

    def choose(self, feats: List[float]) -> int:
        if random.random() < self.epsilon:
            return random.randint(0, 2)
        probs = _softmax([_dot(w, feats) for w in self.weights])
        return int(max(range(3), key=lambda i: probs[i]))

    def prob(self, feats: List[float], action_idx: int) -> float:
        return _softmax([_dot(w, feats) for w in self.weights])[action_idx]

    def update(self, feats: List[float], action: int, reward: float) -> None:
        probs = _softmax([_dot(w, feats) for w in self.weights])
        for a in range(3):
            grad = (1.0 if a == action else 0.0) - probs[a]
            for j in range(self.n_features):
                self.weights[a][j] += self.lr * reward * grad * feats[j]
        self._save()


def build_features(snap: Dict[str, Any], price_change_6h: Optional[float]) -> List[float]:
    """8 features, each in [-1, 1]. Pure function — unit-tested directly.

    [funding_now, funding_z, funding_extreme_signed, oi_change,
     oi_price_divergence, top_position_skew, account_skew, bias]
    """
    f = float(snap.get("funding_rate") or 0.0)
    funding_now = max(-1.0, min(1.0, f / 0.001))          # ±0.1%/8h saturates
    fz = dfx.funding_z(snap.get("funding_hist") or [f])
    extreme = 0.0
    if abs(f) >= config.DERIV_FUNDING_EXTREME:
        extreme = -1.0 if f > 0 else 1.0                   # crowded side gets squeezed

    oi_chg = float(snap.get("oi_change_pct") or 0.0)
    oi_feat = math.tanh(10.0 * oi_chg)

    div = 0.0
    if price_change_6h is not None and abs(oi_chg) > 1e-9:
        oi_up, px_up = oi_chg > 0, price_change_6h > 0
        if oi_up and px_up:
            div = 0.5        # new longs driving price: momentum confirmation
        elif oi_up and not px_up:
            div = -0.5       # new shorts pressing price down
        elif not oi_up and px_up:
            div = -0.25      # short-covering rally: weak
        else:
            div = 0.25       # long liquidation flush: often near exhaustion

    top = float(snap.get("top_position_ratio") or 1.0)
    acct = float(snap.get("global_account_ratio") or 1.0)
    top_skew = math.tanh(math.log(top) if top > 0 else 0.0)
    acct_skew = math.tanh(math.log(acct) if acct > 0 else 0.0)

    return [funding_now, fz, extreme, oi_feat, div, top_skew, acct_skew, 0.2]


class DerivativesAgent:
    def __init__(self, data_fetcher: Optional[Any] = None):
        # spot data fetcher only for the 6h price change (cached OHLCV, free)
        if data_fetcher is None:
            from utils.data_fetcher import DataFetcher
            data_fetcher = DataFetcher()
        self.data = data_fetcher
        self._rl = DerivativesRL()

    def _price_change_6h(self, symbol: str) -> Optional[float]:
        try:
            df = self.data.get_ohlcv(symbol, "1h", limit=10)
            closes = df["close"].astype(float)
            if len(closes) < 7:
                return None
            return float((closes.iloc[-1] - closes.iloc[-7]) / closes.iloc[-7])
        except Exception:
            return None

    def decide(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        unavailable = {"agent": "derivatives_agent", "chartName": symbol,
                       "timeframe": timeframe, "action": "skip",
                       "confidence": 0.0, "available": False, "rl": None}
        try:
            snap = dfx.fetch_derivatives(symbol)
        except Exception:
            snap = None
        if snap is None:
            return unavailable

        feats = build_features(snap, self._price_change_6h(symbol))
        action_idx = self._rl.choose(feats)
        action = _ACTIONS[action_idx]
        conf = float(min(0.9, max(0.5, self._rl.prob(feats, action_idx))))
        return {
            "agent": "derivatives_agent", "chartName": symbol, "timeframe": timeframe,
            "action": action, "confidence": conf, "available": True,
            "details": {
                "funding_rate": snap.get("funding_rate"),
                "oi_change_pct": snap.get("oi_change_pct"),
                "top_position_ratio": snap.get("top_position_ratio"),
                "global_account_ratio": snap.get("global_account_ratio"),
            },
            "rl": {"feats": feats, "action_idx": action_idx},
        }

    def apply_reward(self, feats: Optional[List[float]], action_idx: Optional[int],
                     reward: float) -> None:
        """Stateless RL update from the stored snapshot (same contract as the
        news/research agents — no instance state, no feature race)."""
        if feats is None or action_idx is None:
            return
        self._rl.update(list(feats), int(action_idx), float(reward))


def deriv_note(details: Optional[Dict[str, Any]]) -> Optional[str]:
    """One-line human summary for the Telegram signal message."""
    if not details:
        return None
    f = details.get("funding_rate")
    oi = details.get("oi_change_pct")
    if f is None:
        return None
    parts = [f"funding {f * 100:+.3f}%/8h"]
    if oi is not None:
        parts.append(f"OI {'rising' if oi > 0 else 'falling'} {abs(oi) * 100:.1f}%")
    if f >= config.DERIV_FUNDING_EXTREME and (oi or 0) > 0:
        parts.append("crowded longs — squeeze risk")
    elif f <= -config.DERIV_FUNDING_EXTREME:
        parts.append("crowded shorts — squeeze fuel")
    return ", ".join(parts)
