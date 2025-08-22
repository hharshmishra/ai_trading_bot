# agents/indicator_agent.py
from __future__ import annotations
import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, List

import numpy as np
import pandas as pd
import pandas_ta as ta

from utils.data_fetcher import DataFetcher
from agents import custom_indicators as ci

POLICY_PATH = "logs/indicator_agent_policy.json"
PRED_LOG = "logs/indicator_predictions.jsonl"

# ---------- Small utilities ----------

def _ensure_logs():
    if not os.path.exists("logs"):
        os.makedirs("logs")
    if not os.path.exists(POLICY_PATH):
        with open(POLICY_PATH, "w") as f:
            json.dump({
                # weights are for a simple contextual bandit over two heads:
                #   type1 (direct signals) and type2 (rule-based raw indicators)
                "weights": {"type1": 0.65, "type2": 0.35},
                # track per-signal credibility for multiple direct indicators
                "direct_signals": {},  # e.g., {"nwe": {"weight":0.8,"score":0}}
                "score": 0
            }, f)

def _load_policy():
    if not os.path.exists(POLICY_PATH):
        # Create default policy
        default_policy = {
            "weights": {"type1": 0.65, "type2": 0.35},
            "direct_signals": {},
            "score": 0
        }
        with open(POLICY_PATH, "w") as f:
            json.dump(default_policy, f, indent=4)
        return default_policy
    
    # If file exists but is empty or invalid JSON
    try:
        with open(POLICY_PATH, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        default_policy = {
            "weights": {"type1": 0.65, "type2": 0.35},
            "direct_signals": {},
            "score": 0
        }
        with open(POLICY_PATH, "w") as f:
            json.dump(default_policy, f, indent=4)
        return default_policy


def _save_policy(pol: Dict[str, Any]):
    with open(POLICY_PATH, "w") as f:
        json.dump(pol, f, indent=2)

def _append_jsonl(path: str, obj: Dict[str, Any]):
    with open(path, "a") as f:
        f.write(json.dumps(obj, default=str) + "\n")

# ---------- Data model ----------

@dataclass
class IndicatorDecision:
    agent: str
    chartName: str
    timeframe: str
    action: str            # "buy" | "sell" | "skip"
    confidence: float
    details: Dict[str, Any]

# ---------- Agent ----------

class IndicatorAgent:
    """
    Produces a structured decision from:
    - Type 1: Direct signals (custom indicators like NWE, AlphaTrend, etc.)
    - Type 2: Raw indicator rules (MA ribbon, RSI, MACD, BB position)
    Uses a light RL scheme to adjust weights between Type1 vs Type2 and
    to learn credibility for each direct-signal plugin.
    """

    def __init__(self, prefer_csv: bool = True):
        _ensure_logs()
        self.data = DataFetcher(prefer_csv=prefer_csv)
        self.policy = _load_policy()

    # ----------------- PUBLIC API -----------------

    def decide(self, symbol: str, timeframe: str, ohlcv: Optional[pd.DataFrame] = None,
               limit: int = 500) -> IndicatorDecision:
        """Main entrypoint. Returns a decision and logs it."""
        df = ohlcv if ohlcv is not None else self.data.get_ohlcv(symbol, timeframe, limit=limit)
        df = self._standardize(df)

        # Compute raw indicators needed for Type2
        raw = self._compute_raw_indicators(df)
        type2 = self._type2_rules(raw)

        # Collect Type1 direct signals from custom indicator plugins
        direct_signals = self._collect_direct_signals(df)
        type1 = self._merge_direct_signals(direct_signals)

        # Combine heads using learned weights
        final_action, final_conf, blend_details = self._blend(type1, type2)

        out = IndicatorDecision(
            agent="indicator_agent",
            chartName=symbol, timeframe=timeframe,
            action=final_action,
            confidence=float(np.clip(final_conf, 0.0, 0.999)),
            details={
                "type1": type1,
                "type2": type2,
                "blend": blend_details,
                "direct_signals": direct_signals
            }
        )
        _append_jsonl(PRED_LOG, asdict(out))
        return out

    def learn(self, predicted_action: str, true_outcome: str,
              reward_correct: int = 1, reward_wrong: int = -4):
        """
        RL feedback: update weights depending on whether our final action matched 'true_outcome'.
        You call this AFTER you know the result.
        """
        pol = _load_policy()
        reward = reward_correct if predicted_action == true_outcome else reward_wrong

        # Global score for this agent
        pol["score"] = pol.get("score", 0) + reward

        # Nudge Type1/Type2 weights depending on which contributed more to the last decision.
        # We look at the last line in predictions log to see the blend contributions.
        try:
            with open(PRED_LOG, "r") as f:
                last = None
                for line in f:
                    last = json.loads(line)
            if last and "details" in last and "blend" in last["details"]:
                b = last["details"]["blend"]
                # if we were correct, increase contribution weights that supported the action; else decrease
                sign = 1 if reward > 0 else -1
                pol["weights"]["type1"] = float(np.clip(pol["weights"]["type1"] + sign*0.03*b.get("type1_share", 0.5), 0.05, 0.95))
                pol["weights"]["type2"] = float(np.clip(pol["weights"]["type2"] + sign*0.03*b.get("type2_share", 0.5), 0.05, 0.95))
                # renormalize
                s = pol["weights"]["type1"] + pol["weights"]["type2"]
                pol["weights"]["type1"] = round(pol["weights"]["type1"]/s, 4)
                pol["weights"]["type2"] = round(pol["weights"]["type2"]/s, 4)

                # If a specific direct indicator fired, adapt its credibility
                fired = b.get("fired_direct", None)
                if fired:
                    d = pol["direct_signals"].get(fired, {"weight": 0.7, "score": 0})
                    d["score"] += reward
                    d["weight"] = float(np.clip(d["weight"] + (0.05 if reward > 0 else -0.07), 0.1, 0.95))
                    pol["direct_signals"][fired] = d
        except FileNotFoundError:
            pass

        _save_policy(pol)
        self.policy = pol

    # ----------------- INTERNALS -----------------

    def _standardize(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.rename(columns=str.lower).copy()
        if "timestamp" not in df.columns:
            raise ValueError("OHLCV DataFrame must contain a 'timestamp' column.")
        # Ensure datetime index
        if not np.issubdtype(df["timestamp"].dtype, np.datetime64):
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
        return df[["timestamp","open","high","low","close","volume"]]

    def _compute_raw_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        close = out["close"]

        # Moving averages
        out["ma20"] = close.rolling(20).mean()
        out["ma50"] = close.rolling(50).mean()

        # RSI, MACD, Bollinger
        out["rsi14"] = ta.rsi(close, length=14)
        macd = ta.macd(close)
        out["macd_hist"] = macd["MACDh_12_26_9"]
        bb = ta.bbands(close, length=20, std=2)
        out["bb_lower"] = bb["BBL_20_2.0"]
        out["bb_upper"] = bb["BBU_20_2.0"]
        
        # Stochastic RSI (14 period, fast K=3, fast D=3)
        stochrsi = ta.stochrsi(close, length=14, rsi_length=14, k=3, d=3)
        out["stochrsi_k"] = stochrsi["STOCHRSIk_14_14_3_3"]
        out["stochrsi_d"] = stochrsi["STOCHRSId_14_14_3_3"]
        
        # ✅ SuperTrend (length=10, multiplier=3 is common)
        st = ta.supertrend(out["high"], out["low"], out["close"], length=10, multiplier=3)
        out["supertrend"] = st["SUPERT_10_3.0"]
        out["supertrend_dir"] = st["SUPERTd_10_3.0"]   # 1 = bullish, -1 = bearish

        return out

    def _type2_rules(self, raw: pd.DataFrame) -> Dict[str, Any]:
        r = raw.dropna().iloc[-1]
        votes = {"bull": 0, "bear": 0}

        # MA ribbon
        if r["close"] > r["ma20"] and r["close"] > r["ma50"]:
            votes["bull"] += 2
        elif r["close"] < r["ma20"] and r["close"] < r["ma50"]:
            votes["bear"] += 2

        # RSI extremes
        if r["rsi14"] < 30:
            votes["bull"] += 1
        elif r["rsi14"] > 70:
            votes["bear"] += 1

        # MACD histogram
        if r["macd_hist"] > 0:
            votes["bull"] += 1
        else:
            votes["bear"] += 1

        # BB squeeze-ish positioning bonus (lightweight)
        if r["close"] <= r["bb_lower"]:
            votes["bull"] += 1
        elif r["close"] >= r["bb_upper"]:
            votes["bear"] += 1
        
        # StochRSI (overbought/oversold)
        if r["stochrsi_k"] < 20 and r["stochrsi_d"] < 20:
            votes["bull"] += 1
        elif r["stochrsi_k"] > 80 and r["stochrsi_d"] > 80:
            votes["bear"] += 1
            
        # ✅ SuperTrend direction
        if r["supertrend_dir"] == 1:
            votes["bull"] += 2
        elif r["supertrend_dir"] == -1:
            votes["bear"] += 2

        if votes["bull"] > votes["bear"]:
            action = "buy"
            confidence = 0.55 + 0.1*(votes["bull"] - votes["bear"])
        elif votes["bear"] > votes["bull"]:
            action = "sell"
            confidence = 0.55 + 0.1*(votes["bear"] - votes["bull"])
        else:
            action = "skip"
            confidence = 0.45

        return {
            "action": action,
            "confidence": float(np.clip(confidence, 0.0, 0.98)),
            "votes": votes,
            "last_row": {
                "close": float(r["close"]),
                "ma20": float(r["ma20"]),
                "ma50": float(r["ma50"]),
                "rsi14": float(r["rsi14"]),
                "macd_hist": float(r["macd_hist"]),
                "bb_lower": float(r["bb_lower"]),
                "bb_upper": float(r["bb_upper"]),
                "stochrsi_k": float(r["stochrsi_k"]),
                "stochrsi_d": float(r["stochrsi_d"]),
                "supertrend": float(r["supertrend"]),
                "supertrend_dir": int(r["supertrend_dir"])
            }
        }

    def _collect_direct_signals(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        signals = []
            # Ensure indicator columns exist
        df = ci.apply_nadaraya_watson_envelope(df)
        
        # Example: Nadaraya-Watson envelope direct signal (if you implement it)
        sig = ci.direct_signal_from_nwe(df)
        if sig:
            sig["name"] = "nwe"
            signals.append(sig)

        # You can add more custom direct-signal producers here (AlphaTrend, etc.)
        # --- Chandelier Exit Example ---
        df = ci.chandelier_exit(df)   # adds long_stop, short_stop, ce_signal columns
        latest = df.iloc[-1]
        if latest['ce_signal'] in ["buy", "sell"]:
            signals.append({
                "signal": latest['ce_signal'],
                "confidence": 0.9,  # you can adjust logic to give partial confidence
                "name": "chandelier_exit"
            })
        # Each should return: {"signal":"buy"/"sell"/"skip", "confidence": float, "name": "alpha_trend"}

        return signals

    def _merge_direct_signals(self, signals: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not signals:
            return {"action": "skip", "confidence": 0.5, "used": []}

        pol = self.policy
        # Weighted vote among direct indicators; each indicator has its own learned weight
        scores = {"buy": 0.0, "sell": 0.0, "skip": 0.0}
        used = []

        for s in signals:
            name = s.get("name", "unknown")
            conf = float(s.get("confidence", 0.6))
            # indicator-specific learned weight (credibility)
            w = pol["direct_signals"].get(name, {}).get("weight", 0.7)
            contrib = w * conf
            scores[s["signal"]] += contrib
            used.append({"name": name, "signal": s["signal"], "confidence": conf, "weight": w, "contribution": contrib})

        action = max(scores.items(), key=lambda x: x[1])[0]
        total = sum(scores.values()) if sum(scores.values())>0 else 1.0
        confidence = scores[action] / total

        # record potential top contributor
        fired_direct = None
        if used:
            fired_direct = max(used, key=lambda u: u["contribution"])["name"]

        return {
            "action": action,
            "confidence": float(np.clip(confidence, 0.0, 0.99)),
            "scores": scores,
            "used": used,
            "fired_direct": fired_direct
        }

    def _blend(self, type1: Dict[str, Any], type2: Dict[str, Any]):
        w1 = self.policy["weights"]["type1"]
        w2 = self.policy["weights"]["type2"]

        # Convert actions to directional scores
        def act_to_vec(act: str) -> Dict[str, float]:
            v = {"buy": 0.0, "sell": 0.0, "skip": 0.0}
            v[act] = 1.0
            return v

        s1 = act_to_vec(type1["action"])
        s2 = act_to_vec(type2["action"])

        scores = {
            "buy":  w1 * type1["confidence"] * s1["buy"]  + w2 * type2["confidence"] * s2["buy"],
            "sell": w1 * type1["confidence"] * s1["sell"] + w2 * type2["confidence"] * s2["sell"],
            "skip": w1 * (1.0 - type1["confidence"]) * s1["skip"] + w2 * (1.0 - type2["confidence"]) * s2["skip"]
        }

        action = max(scores.items(), key=lambda x: x[1])[0]
        tot = sum(scores.values()) if sum(scores.values())>0 else 1.0
        confidence = scores[action] / tot

        blend_details = {
            "type1_weight": w1, "type2_weight": w2,
            "type1_share": float(w1 / (w1 + w2)),
            "type2_share": float(w2 / (w1 + w2)),
            "scores": scores,
            "fired_direct": type1.get("fired_direct")
        }
        return action, confidence, blend_details
