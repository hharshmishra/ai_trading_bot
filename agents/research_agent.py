# agents/research_agent.py
# -----------------------------------------------------------------------------
# ResearchAgent: reinforcement-learning agent that models crypto market context
# and fundamentals to assist final trade decision. It implements:
#   Logic 1  – Ecosystem regime (ETH, SOL, BSC, etc.) via drivers + news + open API
#   Logic 2  – US equities (S&P 500 / SPX) influence via news
#   Logic 3  – Money-flow phase (BTC → ETH → Large Caps → Small Caps/Memes)
#   Logic 4  – Bitcoin Dominance effect (BTCDOMUSDT) on Alts
#   Logic 5  – DXY inverse relation via news
#
# Features from each logic are blended into a contextual bandit (3 actions:
# sell, skip, buy). You can call .learn(+1 / -4) after the outcome.
#
# Dependencies (already present in your project):
#   - utils.data_fetcher.DataFetcher (CSV-first; CCXT fallback)
#   - agents.indicator_agent.IndicatorAgent (for Type-1/Type-2 trend summaries)
#   - agents.news_agent.NewsAgent (optional; requires OPENAI_API_KEY)
#
# How to use:
#   agent = ResearchAgent()
#   out = agent.decide("ETCUSDT", "1h", indicator_agent=IndicatorAgent(), news_agent=NewsAgent())
#   print(out)
#   agent.learn(predicted_action=out["action"], true_outcome="buy")  # +1 if correct else -4
# -----------------------------------------------------------------------------
from __future__ import annotations
import os, json, math, time, random
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import asdict, is_dataclass
import numpy as np
import pandas as pd

from utils.data_fetcher import DataFetcher

# ---------------- Configuration -----------------
POLICY_PATH = "logs/research_agent_policy.json"
PRED_LOG    = "logs/research_predictions.jsonl"

# Ecosystem map (editable). Symbols should be BASE asset tickers (no /USDT).
ECOSYSTEMS: Dict[str, List[str]] = {
    'ethereum': ['ETH', 'ETC', 'LINK', 'UNI', 'AAVE', 'COMP', 'MKR', 'SNX', 'LRC', 'MATIC'],
    'binance_smart_chain': ['BNB', 'CAKE', 'AUTO', 'BUNNY', 'BIFI'],
    'solana': ['SOL', 'RAY', 'SRM', 'FIDA', 'ROPE'],
    'cardano': ['ADA', 'COTI'],
    'polkadot': ['DOT', 'KSM', 'OCEAN', 'AKRO'],
    'avalanche': ['AVAX', 'PNG', 'JOE'],
    'cosmos': ['ATOM', 'OSMO', 'JUNO', 'SCRT'],
    'defi': ['LINK', 'UNI', 'AAVE', 'COMP', 'MKR', 'SNX', 'CRV', 'BAL', '1INCH', 'PENDLE', 'DYDX'],
    'layer2': ['MATIC', 'LRC', 'IMX', 'METIS', 'ARB', 'OP', 'POL', 'MANTA'],
    'gaming': ['AXS', 'SLP', 'SAND', 'MANA', 'ENJ', 'ALICE', 'GALA'],
    'nft': ['FLOW', 'ENJ', 'SAND', 'MANA', 'THETA'],
    'oracle': ['LINK', 'BAND', 'TRB', 'PYTH'],
    'exchange_tokens': ['BNB', 'FTT', 'KCS', 'GT', 'HT'],
    'ai': ['FET', 'RNDR', 'INJ', 'GRT', 'WLD'],
    'meme': ['DOGE', 'SHIB'],
    'layer1': ['BTC', 'BCH', 'LUNA', 'NEAR', 'ICP', 'APT'],
    'storage': ['FIL', 'AR', 'STORJ'],
    'social': ['GMT', 'ARKM'],
    'iot': ['HNT'],
    'payments': ['XRP', 'WAVES']
}

# For Logic 1 (ecosystem), define up to 3 driver coins per ecosystem in priority order.
ECOSYSTEM_DRIVERS: Dict[str, List[str]] = {
    'ethereum': ['ETH', 'MATIC', 'LINK'],
    'solana': ['SOL', 'JUP', 'PYTH'],
    'binance_smart_chain': ['BNB', 'CAKE'],
    'polkadot': ['DOT', 'KSM'],
    'avalanche': ['AVAX'],
    'cosmos': ['ATOM', 'OSMO'],
    'defi': ['LINK', 'AAVE', 'UNI'],
    'layer2': ['MATIC', 'ARB', 'OP'],
    'ai': ['FET', 'RNDR', 'INJ'],
    'meme': ['DOGE', 'SHIB'],
}

# Optional: an external sentiment API base URL (if you have one). Leave None to disable.
OPEN_SENTIMENT_API_BASE = None  # e.g., "https://your-svc/api"

# ---------------- Small utilities -----------------
def structure_output(indicator_decision):
    """
    Convert IndicatorDecision object into a clean structured dictionary/JSON.
    """
    # If your IndicatorDecision is a dataclass, use asdict()
    if is_dataclass(indicator_decision):
        data = asdict(indicator_decision)
    else:
        # Fallback: pull attributes manually
        data = {
            "agent": indicator_decision.agent,
            "chartName": indicator_decision.chartName,
            "timeframe": indicator_decision.timeframe,
            "action": indicator_decision.action,
            "confidence": indicator_decision.confidence,
            "details": indicator_decision.details,
            "blend": indicator_decision.blend,
            "direct_signals": indicator_decision.direct_signals,
        }
    
    # Pretty JSON
    return json.dumps(data, indent=4)

def _ensure_logs():
    os.makedirs("logs", exist_ok=True)

def _strip_suffix(symbol: str) -> str:
    s = (symbol or "").upper().replace("/", "")
    for suf in ["USDT","USDC","USD","INR","BUSD","FDUSD","TUSD"]:
        if s.endswith(suf):
            return s[:-len(suf)]
    return s

# Scores

def _sent_to_score(sent: Optional[str], conf: Optional[float]) -> float:
    s = (sent or "").lower()
    c = float(conf or 0.5)
    if s.startswith("bull"): return  c
    if s.startswith("bear"): return -c
    return 0.0

def _action_to_score(action: Optional[str], conf: Optional[float]) -> float:
    a = (action or "").lower()
    c = float(conf or 0.5)
    if a == "buy":  return  c
    if a == "sell": return -c
    return 0.0

# ---------------- RL policy (contextual bandit) -----------------

@dataclass
class BanditPolicy:
    weights: List[List[float]]  # 3 x F (sell, skip, buy)
    epsilon: float

    @staticmethod
    def default(n_features: int) -> "BanditPolicy":
        rng = random.Random(42)
        weights = [[rng.uniform(-0.05, 0.05) for _ in range(n_features)] for _ in range(3)]
        return BanditPolicy(weights=weights, epsilon=0.08)


def _dot(a: List[float], b: List[float]) -> float:
    return float(sum(x*y for x,y in zip(a,b)))


def _softmax(logits: List[float]) -> List[float]:
    m = max(logits)
    exps = [math.exp(l - m) for l in logits]
    s = sum(exps)
    return [e/s for e in exps]


class ResearchRL:
    def __init__(self, n_features: int, lr: float = 0.05):
        self.n_features = n_features
        self.lr = lr
        pol = self._load()
        if pol is None:
            self.policy = BanditPolicy.default(n_features)
            self._save()
        else:
            self.policy = pol

    def _load(self) -> Optional[BanditPolicy]:
        try:
            if os.path.exists(POLICY_PATH) and os.path.getsize(POLICY_PATH) > 0:
                with open(POLICY_PATH, "r", encoding="utf-8") as f:
                    p = json.load(f)
                return BanditPolicy(weights=p["weights"], epsilon=p.get("epsilon", 0.08))
        except Exception:
            return None
        return None

    def _save(self):
        with open(POLICY_PATH, "w", encoding="utf-8") as f:
            json.dump({"weights": self.policy.weights, "epsilon": self.policy.epsilon}, f, indent=2)

    def choose(self, feats: List[float]) -> int:
        if random.random() < self.policy.epsilon:
            return random.randint(0,2)
        logits = [_dot(w, feats) for w in self.policy.weights]
        probs  = _softmax(logits)
        return int(max(range(3), key=lambda i: probs[i]))

    def update(self, feats: List[float], action: int, reward: float):
        logits = [_dot(w, feats) for w in self.policy.weights]
        probs  = _softmax(logits)
        for a in range(3):
            grad_coeff = (1.0 if a==action else 0.0) - probs[a]
            for j in range(self.n_features):
                self.policy.weights[a][j] += self.lr * reward * grad_coeff * feats[j]
        self._save()

# ---------------- Main Agent -----------------

class ResearchAgent:
    """
    Reinforcement-learning research agent implementing 5 logics.

    Features vector (F=10):
      [eco_score, eco_news, eco_ind, spx_news, money_flow_phase, btdom_effect,
       dxy_news, child_trend, parent_trend, bias]
    Each feature is in [-1, 1].
    """
    def __init__(self, prefer_csv: bool = False):
        _ensure_logs()
        self.data = DataFetcher(prefer_csv=prefer_csv)
        self._rl = ResearchRL(n_features=10)
        self._last_feats: Optional[List[float]] = None
        self._last_action: Optional[int] = None

    # ---------- Public API ----------
    def decide(self,
               symbol: str,
               timeframe: str,
               indicator_agent: Optional[Any] = None,
               news_agent: Optional[Any] = None,
               limit: int = 500) -> Dict[str, Any]:
        """
        Returns a dict: {
           agent, chartName, timeframe, action, confidence, details
        }
        """
        base = _strip_suffix(symbol)
        child_df = self.data.get_ohlcv(symbol, timeframe, limit=limit)

        # Parent proxy: if symbol is an ALT (not BTC), try BTC or ecosystem leader
        parent_symbol = self._select_parent_for(base)
        parent_df = None
        if parent_symbol:
            try:
                parent_df = self.data.get_ohlcv(parent_symbol, timeframe, limit=limit)
            except Exception:
                parent_df = None

        # Compute features from 5 logics
        eco_score, eco_news, eco_ind, eco_details = self._logic1_ecosystem(base, timeframe, indicator_agent, news_agent)
        spx_news, spx_details = self._logic2_spx(news_agent)
        money_flow, mf_details = self._logic3_money_flow(timeframe, indicator_agent, news_agent)
        btdom_effect, btd_details = self._logic4_btcdominance(timeframe, indicator_agent, news_agent)
        dxy_news, dxy_details = self._logic5_dxy(news_agent)

        child_trend = self._child_trend(child_df)
        parent_trend = self._parent_trend(parent_df, indicator_agent, timeframe)
        bias = 0.2 if symbol.upper().endswith("USDT") else 0.0

        feats = [eco_score, eco_news, eco_ind, spx_news, money_flow,
                 btdom_effect, dxy_news, child_trend, parent_trend, bias]

        action_idx = self._rl.choose(feats)
        action = ["sell","skip","buy"][action_idx]

        # Confidence from magnitude of winning logit
        w = self._rl.policy.weights[action_idx]
        mag = abs(sum(float(wi)*float(xi) for wi,xi in zip(w, feats)))
        confidence = float(1.0 / (1.0 + math.exp(-2.5 * mag)))

        self._last_feats = feats
        self._last_action = action_idx

        details = {
            "features": {
                "eco_score": eco_score,
                "eco_news": eco_news,
                "eco_ind": eco_ind,
                "spx_news": spx_news,
                "money_flow_phase": money_flow,
                "btcdominance_effect": btdom_effect,
                "dxy_news": dxy_news,
                "child_trend": child_trend,
                "parent_trend": parent_trend,
                "bias": bias
            },
            "logic_details": {
                "logic1": eco_details,
                "logic2": spx_details,
                "logic3": mf_details,
                "logic4": btd_details,
                "logic5": dxy_details,
            }
        }

        row = {
            "ts": pd.Timestamp.utcnow().isoformat(),
            "symbol": symbol, "timeframe": timeframe,
            "action": action, "confidence": round(confidence,4),
            "feats": feats,
        }
        with open(PRED_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

        return {
            "agent": "research_agent",
            "chartName": symbol,
            "timeframe": timeframe,
            "action": action,
            "confidence": round(confidence,4),
            "details": details
        }

    def learn(self, predicted_action: str | int, true_outcome: str | float | int | None = None, reward: Optional[float] = None):
        if self._last_feats is None or self._last_action is None:
            return
        if reward is None and true_outcome is not None:
            try:
                outcome = str(true_outcome).lower()
                pred = predicted_action if isinstance(predicted_action, str) else ["sell","skip","buy"][int(predicted_action)]
                reward = 1.0 if outcome == pred else -4.0
            except Exception:
                reward = -1.0
        elif reward is None:
            reward = -1.0
        self._rl.update(self._last_feats, self._last_action, float(reward))

    # ---------- Logic 1: Ecosystem regime ----------
    def _logic1_ecosystem(self, base: str, timeframe: str, indicator_agent: Optional[Any], news_agent: Optional[Any]) -> Tuple[float,float,float,Dict[str,Any]]:
        ecos = self._ecos_for_asset(base)
        if not ecos:
            return 0.0, 0.0, 0.0, {"ecosystems": []}

        # Pick primary ecosystem (first hit) and its drivers
        eco = ecos[0]
        drivers = ECOSYSTEM_DRIVERS.get(eco, ECOSYSTEMS.get(eco, [])[:3])
        drivers = [d for d in drivers if d != base]
        driver_pairs = [d + "USDT" for d in drivers]
        # Indicator-based leader trend (avg of driver actions)
        ind_scores = []
        ind_raw = []
        if indicator_agent is not None:
            for dp in driver_pairs[:3]:
                try:
                    out = indicator_agent.decide(dp, timeframe)
                    act = getattr(out, 'action', None) if hasattr(out, 'action') else out.get('action')
                    conf = float(getattr(out, 'confidence', 0.6)) if hasattr(out, 'confidence') else float(out.get('confidence', 0.6))
                    s = _action_to_score(act, conf)
                    ind_scores.append(s)
                    ind_raw.append({"pair": dp, "action": act, "confidence": conf})
                except Exception as e:
                    print(f"ERROR for {dp}: {e}")
        eco_ind = float(np.tanh(np.mean(ind_scores))) if ind_scores else 0.0


        # News-based ecosystem sentiment (via drivers + optional open API)
        news_scores = []
        news_raw = []
        if news_agent is not None:
            for dp in driver_pairs[:3]:
                try:
                    r = news_agent.run(pair=dp)
                    pj = r.get("pair_json", {})
                    s = _sent_to_score(pj.get("sentiment"), pj.get("confidence"))
                    news_scores.append(s)
                    news_raw.append({"pair": dp, "sentiment": pj.get("sentiment"), "confidence": pj.get("confidence")})
                except Exception:
                    pass
        # Optional external sentiment API for the ecosystem name
        ext_sent = None
        if OPEN_SENTIMENT_API_BASE:
            try:
                import requests
                q = eco.replace("_", "%20")
                url = f"{OPEN_SENTIMENT_API_BASE}/sentiment?ecosystem={q}"
                ext = requests.get(url, timeout=5).json()
                # Expect ext = {"sentiment": "Bullish|Bearish|Neutral", "confidence": 0.xx}
                ext_sent = _sent_to_score(ext.get("sentiment"), ext.get("confidence"))
                news_scores.append(ext_sent)
            except Exception:
                ext_sent = None
        eco_news = float(np.tanh(np.mean(news_scores))) if news_scores else 0.0

        # Ecosystem score combines news + indicator with small preference to indicators
        eco_score = float(np.clip(0.6*eco_ind + 0.4*eco_news, -1.0, 1.0))

        return eco_score, eco_news, eco_ind, {
            "ecosystems": ecos,
            "primary": eco,
            "drivers": driver_pairs[:3],
            "indicator_driver_votes": ind_raw,
            "news_driver_votes": news_raw,
            "external_ecosystem_sent": ext_sent
        }

    def _ecos_for_asset(self, base: str) -> List[str]:
        base = base.upper()
        hits = []
        for eco, members in ECOSYSTEMS.items():
            if base in [m.upper() for m in members]:
                hits.append(eco)
        return hits

    def _select_parent_for(self, base: str) -> Optional[str]:
        b = base.upper()
        if b == "BTC":
            return None
        # If asset belongs to an ecosystem with a clear leader, use that as parent
        ecos = self._ecos_for_asset(b)
        if ecos:
            eco = ecos[0]
            drivers = ECOSYSTEM_DRIVERS.get(eco, ECOSYSTEMS.get(eco, [])[:1])
            if drivers:
                leader = drivers[0]
                if leader == b:
                    return "BTCUSDT"  # fallback to BTC when the leader is itself
                return leader + "USDT"
        # Default parent: BTC
        return "BTCUSDT"

    # ---------- Logic 2: SPX influence via news ----------
    def _logic2_spx(self, news_agent: Optional[Any]) -> Tuple[float, Dict[str, Any]]:
        if news_agent is None:
            return 0.0, {"used": False}
        try:
            r = news_agent.run(pair="SPX")  # let your NewsAgent map this to S&P 500
            oj = r.get("overall_json", {})
            pj = r.get("pair_json", {})
            s = _sent_to_score(pj.get("sentiment") or oj.get("sentiment"), pj.get("confidence") or oj.get("confidence"))
            return float(np.clip(s, -1, 1)), {"used": True, "pair_json": pj, "overall_json": oj}
        except Exception:
            return 0.0, {"used": False}

    # ---------- Logic 3: Money-flow phase ----------
    def _logic3_money_flow(self, timeframe: str, indicator_agent: Optional[Any], news_agent: Optional[Any]) -> Tuple[float, Dict[str, Any]]:
        """
        Heuristic phase computation using readily available pairs on Binance:
          - BTCUSDT trend, ETHUSDT trend, ETHBTC trend, a basket of large-cap alts.
        Score in [-1, 1]: negative favors BTC dominance phase; positive favors ALTs/small caps.
        """
        pairs_btc = ["BTCUSDT"]
        pairs_eth = ["ETHUSDT", "ETHBTC"]
        alt_basket = ["LINKUSDT", "MATICUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]

        votes = []
        def trend_of(pair):
            try:
                if indicator_agent is None:
                    return 0.0
                out = indicator_agent.decide(pair, timeframe)
                act = getattr(out, 'action', None) if hasattr(out, 'action') else out.get('action')
                conf = float(getattr(out, 'confidence', 0.6)) if hasattr(out, 'confidence') else float(out.get('confidence', 0.6))
                return _action_to_score(act, conf)
            except Exception:
                return 0.0

        btc_tr = np.mean([trend_of(p) for p in pairs_btc])
        eth_tr = np.mean([trend_of(p) for p in pairs_eth])
        alt_tr = np.mean([trend_of(p) for p in alt_basket])

        # Phase scoring: BTC -> ETH -> ALTS progression
        # Map to [-1, 1]: -1 early BTC phase, 0 mid ETH phase, +1 ALT phase
        phase_raw = 0.0
        if btc_tr > 0.3 and eth_tr < 0.1 and alt_tr < 0.1:
            phase_raw = -0.8  # BTC phase
        elif eth_tr > 0.3 and alt_tr < 0.1:
            phase_raw = -0.2  # ETH phase starting
        elif alt_tr > 0.3:
            phase_raw = 0.7   # ALT phase
        else:
            # soft blend
            phase_raw = 0.3*alt_tr + 0.1*eth_tr - 0.2*btc_tr
        return float(np.clip(phase_raw, -1, 1)), {
            "btc_trend": float(btc_tr),
            "eth_trend": float(eth_tr),
            "alt_basket_trend": float(alt_tr)
        }

    # ---------- Logic 4: Bitcoin dominance ----------
    def _logic4_btcdominance(self, timeframe: str, indicator_agent: Optional[Any], news_agent: Optional[Any]) -> Tuple[float, Dict[str, Any]]:
        """
        If BTC up and BTCDOM up → ALTs fall (negative for alts).
        If BTC down and BTCDOM down → ALTs rise (positive for alts).
        Score positive means favorable to ALT longs; negative means unfavorable to ALT longs.
        """
        btc_score = 0.0
        dom_score = 0.0
        details = {}

        if indicator_agent is not None:
            try:
                b = indicator_agent.decide("BTCUSDT", timeframe)
                btc_score = _action_to_score(getattr(b,'action', None), getattr(b,'confidence', 0.6))
            except Exception as e:
                print(e)
            try:
                d = indicator_agent.decide("BTCDOMUSDT", timeframe)
                dom_score = _action_to_score(getattr(d,'action', None), getattr(d,'confidence', 0.6))
            except Exception as e:
                print(e)

        # Convert to ALT favorability per your rule
        alt_favor = 0.0
        if btc_score > 0 and dom_score > 0:
            alt_favor = -0.7
        elif btc_score < 0 and dom_score < 0:
            alt_favor = 0.7
        else:
            alt_favor = 0.2*(-btc_score) + 0.2*(-dom_score)
        details.update({"btc_score": btc_score, "btcdom_score": dom_score})

        return float(np.clip(alt_favor, -1, 1)), details

    # ---------- Logic 5: DXY inverse via news ----------
    def _logic5_dxy(self, news_agent: Optional[Any]) -> Tuple[float, Dict[str, Any]]:
        if news_agent is None:
            return 0.0, {"used": False}
        try:
            r = news_agent.run(pair="DXY")
            oj = r.get("overall_json", {})
            pj = r.get("pair_json", {})
            s = _sent_to_score(pj.get("sentiment") or oj.get("sentiment"), pj.get("confidence") or oj.get("confidence"))
            # Inverse relation: bullish DXY → bearish BTC → negative score
            return float(np.clip(-s, -1, 1)), {"used": True, "pair_json": pj, "overall_json": oj}
        except Exception:
            return 0.0, {"used": False}

    # ---------- Helper trends ----------
    def _child_trend(self, df: Optional[pd.DataFrame]) -> float:
        if df is None or df.empty:
            return 0.0
        x = df.copy()
        x["ma50"] = x["close"].rolling(50, min_periods=1).mean()
        x["ma200"] = x["close"].rolling(200, min_periods=1).mean()
        last = x.iloc[-1]
        score = 0.0
        score += 0.5 if last["close"] > last["ma50"] else -0.5
        score += 0.5 if last["close"] > last["ma200"] else -0.5
        return float(np.clip(score, -1, 1))

    def _parent_trend(self, parent_df: Optional[pd.DataFrame], indicator_agent: Optional[Any], timeframe: str) -> float:
        if parent_df is None:
            return 0.0
        # Prefer indicator agent signal on the parent if available; else MA regime.
        try:
            if indicator_agent is not None:
                # We don't know parent symbol here; parent_df was fetched already just to be safe.
                # So fallback to MA regime on parent_df to avoid refetch.
                x = parent_df.copy()
                x["ma50"] = x["close"].rolling(50, min_periods=1).mean()
                x["ma200"] = x["close"].rolling(200, min_periods=1).mean()
                last = x.iloc[-1]
                score = 0.0
                score += 0.5 if last["close"] > last["ma50"] else -0.5
                score += 0.5 if last["close"] > last["ma200"] else -0.5
                return float(np.clip(score, -1, 1))
        except Exception:
            pass
        return 0.0