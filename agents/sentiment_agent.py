"""SentimentAgent — 5th brain voter (v3.5). No LLM, $0 data.

Turns crowd state into a learned vote: Fear & Greed level/trend/extremes
(contrarian fuel at the tails — every major BTC bottom printed prolonged
extreme fear), BTC on-chain activity (mempool fee pressure ranked the #1
predictor class in recent ML studies; tx momentum; price-vs-usage divergence
— the hollow-rally detector), retail attention (CoinGecko trending), and
per-pair taker buy/sell order-flow (field 9 of raw Binance klines — the
short-horizon driver the unified OHLCV path throws away).

Same 3-action linear softmax bandit as research/derivatives. Any total data
outage returns ``available: False`` with confidence 0.0 — a mathematical
no-op in the brain's weighted sum; NULL feats mean the grader applies no
reward. The bandit owns the SIGN of every feature: the literature justifies
inclusion, live graded outcomes decide direction and size.
"""
from __future__ import annotations

import json
import math
import os
import random
from typing import Any, Dict, List, Optional, Tuple

import config
from utils import sentiment_fetcher as sfx

POLICY_PATH = "logs/sentiment_agent_policy.json"

N_FEATURES = 10
WEIGHT_CLAMP = 5.0   # v3.7: bandit weights stay in [-5, +5]
_ACTIONS = ["sell", "skip", "buy"]


def _softmax(logits: List[float]) -> List[float]:
    m = max(logits)
    exps = [math.exp(v - m) for v in logits]
    s = sum(exps)
    return [e / s for e in exps]


def _dot(a: List[float], b: List[float]) -> float:
    return float(sum(x * y for x, y in zip(a, b)))


class SentimentRL:
    """3 x N_FEATURES linear softmax bandit (mirrors DerivativesRL)."""

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
                w = self.weights[a][j] + self.lr * reward * grad * feats[j]
                self.weights[a][j] = max(-WEIGHT_CLAMP, min(WEIGHT_CLAMP, w))
        self._save()


def _z_last(series: Optional[List[float]]) -> float:
    """z of the last value vs the rest, tanh-squashed to [-1, 1]."""
    if not series or len(series) < 8:
        return 0.0
    cur, hist = series[-1], series[:-1]
    mean = sum(hist) / len(hist)
    var = sum((x - mean) ** 2 for x in hist) / max(len(hist) - 1, 1)
    sd = math.sqrt(var)
    if sd <= 1e-12:
        return 0.0
    return math.tanh((cur - mean) / sd / 2.0)


def _roc(series: Optional[List[float]], back_frac: float = 0.25) -> Optional[float]:
    """Fractional change of the last value vs ~back_frac of the series ago."""
    if not series or len(series) < 8:
        return None
    idx = max(0, len(series) - 1 - int(len(series) * back_frac))
    base = series[idx]
    if abs(base) <= 1e-12:
        return None
    return (series[-1] - base) / abs(base)


def build_features(bundle: Optional[Dict[str, Any]],
                   flow: Optional[List[Tuple[float, float, float]]],
                   btc_roc_7d: Optional[float],
                   base_ticker: str = "") -> List[float]:
    """10 features, each in [-1, 1]. Pure function — unit-tested directly.

    [fng_level, fng_roc, fng_extreme, fee_pressure, tx_momentum,
     onchain_divergence, trending_hit, taker_ratio_z, taker_trend, bias]
    Missing inputs degrade to 0.0 feature-by-feature.
    """
    b = bundle or {}

    fng = b.get("fng")
    fng_level = (float(fng) - 50.0) / 50.0 if fng is not None else 0.0

    fng_roc = 0.0
    hist = b.get("fng_hist")
    if hist and len(hist) >= 8:
        fng_roc = max(-1.0, min(1.0, (hist[-1] - hist[-8]) / 25.0))

    fng_extreme = 0.0
    if fng is not None:
        f = float(fng)
        if f <= 20.0:
            fng_extreme = (20.0 - f) / 20.0        # deep fear = contrarian long fuel
        elif f >= 80.0:
            fng_extreme = -(f - 80.0) / 20.0       # euphoria = contrarian short fuel

    fee_pressure = _z_last(b.get("mempool"))

    tx_roc = _roc(b.get("ntx"))
    tx_momentum = math.tanh(3.0 * tx_roc) if tx_roc is not None else 0.0

    onchain_div = 0.0
    vol_roc = _roc(b.get("txvol"))
    if btc_roc_7d is not None and vol_roc is not None:
        # price outrunning usage = hollow rally (negative); usage outrunning
        # price = silent accumulation (positive). NVT-flavored.
        onchain_div = math.tanh(3.0 * (vol_roc - btc_roc_7d))

    trending = b.get("trending")
    trending_hit = 1.0 if (trending and base_ticker.upper() in trending) else 0.0

    taker_ratio_z, taker_trend = 0.0, 0.0
    if flow and len(flow) >= 12:
        ratios = [(tb / v if v > 0 else 0.5) for _, v, tb in flow]
        centered = [2.0 * (r - 0.5) for r in ratios]
        taker_ratio_z = _z_last([r for r in ratios])
        recent = centered[-5:]
        prior = centered[:-5]
        taker_trend = math.tanh(3.0 * (sum(recent) / len(recent)
                                       - sum(prior) / len(prior)))

    return [fng_level, fng_roc, fng_extreme, fee_pressure, tx_momentum,
            onchain_div, trending_hit, taker_ratio_z, taker_trend, 0.2]


class SentimentAgent:
    def __init__(self, data_fetcher: Optional[Any] = None):
        # spot fetcher only for the BTC 7d ROC (cached OHLCV, free)
        if data_fetcher is None:
            from utils.data_fetcher import DataFetcher
            data_fetcher = DataFetcher()
        self.data = data_fetcher
        self._rl = SentimentRL()

    def _btc_roc_7d(self) -> Optional[float]:
        try:
            df = self.data.get_ohlcv("BTCUSDT", "1d", limit=10)
            closes = df["close"].astype(float)
            if len(closes) < 8:
                return None
            return float((closes.iloc[-1] - closes.iloc[-8]) / closes.iloc[-8])
        except Exception:
            return None

    def decide(self, symbol: str, timeframe: str,
               market_context: Optional[Any] = None) -> Dict[str, Any]:
        unavailable = {"agent": "sentiment_agent", "chartName": symbol,
                       "timeframe": timeframe, "action": "skip",
                       "confidence": 0.0, "available": False, "rl": None}
        try:
            bundle = sfx.fetch_market_sentiment()
        except Exception:
            bundle = None
        try:
            flow = sfx.fetch_taker_flow(symbol, timeframe)
        except Exception:
            flow = None
        if bundle is None and flow is None:
            return unavailable

        base = symbol[:-4] if symbol.upper().endswith("USDT") else symbol
        feats = build_features(bundle, flow, self._btc_roc_7d(), base_ticker=base)
        action_idx = self._rl.choose(feats)
        action = _ACTIONS[action_idx]
        conf = float(min(0.9, max(0.5, self._rl.prob(feats, action_idx))))
        b = bundle or {}
        taker_pct = None
        if flow:
            _, v, tb = flow[-1]
            taker_pct = (tb / v) if v > 0 else None
        return {
            "agent": "sentiment_agent", "chartName": symbol, "timeframe": timeframe,
            "action": action, "confidence": conf, "available": True,
            "details": {
                "fng": b.get("fng"),
                "fee_pressure_z": feats[3],
                "onchain_divergence": feats[5],
                "trending_hit": bool(feats[6]),
                "taker_buy_pct": taker_pct,
            },
            "rl": {"feats": feats, "action_idx": action_idx},
        }

    def apply_reward(self, feats: Optional[List[float]], action_idx: Optional[int],
                     reward: float) -> None:
        """Stateless RL update from the stored snapshot (news/research/deriv
        contract — no instance state, no feature race)."""
        if feats is None or action_idx is None:
            return
        self._rl.update(list(feats), int(action_idx), float(reward))


def sentiment_note(details: Optional[Dict[str, Any]]) -> Optional[str]:
    """One-line human summary for the Telegram signal message."""
    if not details:
        return None
    parts = []
    fng = details.get("fng")
    if fng is not None:
        f = float(fng)
        label = ("extreme fear" if f <= 20 else "fear" if f <= 40 else
                 "neutral" if f < 60 else "greed" if f < 80 else "extreme greed")
        parts.append(f"F&G {f:.0f} ({label})")
    tp = details.get("taker_buy_pct")
    if tp is not None:
        parts.append(f"taker {tp * 100:.0f}% buy")
    fee = details.get("fee_pressure_z")
    if fee is not None and abs(fee) >= 0.5:
        parts.append("fees hot" if fee > 0 else "fees quiet")
    if details.get("trending_hit"):
        parts.append("trending on CoinGecko")
    return ", ".join(parts) if parts else None
