# agents/indicator_agent.py
from __future__ import annotations
import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, List

import numpy as np
import pandas as pd
import pandas_ta as ta

import config
from utils.data_fetcher import DataFetcher
from agents import custom_indicators as ci
from agents.regime_agent import classify_regime

POLICY_PATH = "logs/indicator_agent_policy.json"
PRED_LOG = "logs/indicator_predictions.jsonl"

_DIRECT_CONF_CACHE = {"mtime": None, "conf": {}}


def _direct_conf(name: str, default: float) -> float:
    """Empirical-Bayes confidence for a direct signal (enhancement D4), from
    the nightly artifact — replaces the hardcoded 0.9s that treated every
    Chandelier/AlphaTrend flip as near-certain regardless of track record.
    mtime-cached; missing artifact / flag off / unknown name -> default."""
    if not config.EMPIRICAL_DIRECT_CONF:
        return default
    path = config.INDICATOR_CONF_PATH
    try:
        mtime = os.path.getmtime(path)
        if _DIRECT_CONF_CACHE["mtime"] != mtime:
            with open(path) as f:
                _DIRECT_CONF_CACHE["conf"] = (json.load(f).get("conf") or {})
            _DIRECT_CONF_CACHE["mtime"] = mtime
        entry = _DIRECT_CONF_CACHE["conf"].get(name)
        return float(entry["conf"]) if entry else default
    except Exception:
        return default


def supertrend_flip_from_raw(raw: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """Trend trigger: SuperTrend direction flip on the last closed bar.

    Reuses the SUPERT columns already computed for the type-2 rules — same
    parameters, zero drift between the trigger and the vote. Confidence scales
    with the close's distance from the flipped supertrend line in ATR units.
    """
    if raw is None or "supertrend_dir" not in raw.columns or len(raw) < 2:
        return None
    d = raw.dropna(subset=["supertrend_dir"])
    if len(d) < 2:
        return None
    cur = int(d["supertrend_dir"].iloc[-1])
    prev = int(d["supertrend_dir"].iloc[-2])
    if cur == prev or cur not in (1, -1):
        return None
    close = float(d["close"].iloc[-1])
    line = float(d["supertrend"].iloc[-1]) if "supertrend" in d.columns else close
    atr = None
    if "atr" in d.columns and pd.notna(d["atr"].iloc[-1]):
        atr = float(d["atr"].iloc[-1])
    conf = 0.65
    if atr and atr > 0:
        conf += 0.25 * min(abs(close - line) / atr, 1.0)
    return {"signal": "buy" if cur == 1 else "sell",
            "confidence": float(np.clip(conf, 0.5, 0.95)),
            "name": "supertrend_flip"}

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
                "type2_rules": {},     # per-rule type-2 credibility (v3.4)
                "score": 0
            }, f)

def _load_policy():
    if not os.path.exists(POLICY_PATH):
        # Create default policy
        default_policy = {
            "weights": {"type1": 0.65, "type2": 0.35},
            "direct_signals": {},
            "type2_rules": {},
            "score": 0
        }
        with open(POLICY_PATH, "w") as f:
            json.dump(default_policy, f, indent=4)
        return default_policy
    
    # If file exists but is empty or invalid JSON
    try:
        with open(POLICY_PATH, "r") as f:
            pol = json.load(f)
        pol.setdefault("type2_rules", {})   # additive schema migration (v3.4)
        return pol
    except (json.JSONDecodeError, FileNotFoundError):
        default_policy = {
            "weights": {"type1": 0.65, "type2": 0.35},
            "direct_signals": {},
            "type2_rules": {},
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

    def __init__(self, prefer_csv: bool = False,
                 nwe_h: float = 8.0, nwe_mult: float = 3.0):
        _ensure_logs()
        self.data = DataFetcher(prefer_csv=prefer_csv)
        self.policy = _load_policy()
        # NWE kernel params — production defaults; the backtest sweep
        # constructs agents with candidate values.
        self.nwe_h = float(nwe_h)
        self.nwe_mult = float(nwe_mult)

    # ----------------- PUBLIC API -----------------

    def decide(self, symbol: str, timeframe: str, ohlcv: Optional[pd.DataFrame] = None,
               limit: int = 500, log: bool = True) -> IndicatorDecision:
        """Main entrypoint. Returns a decision and logs it.

        ``log=False`` suppresses the JSONL append — used by the backtest
        engine, which replays this exact function thousands of times per pair.
        """
        df = ohlcv if ohlcv is not None else self.data.get_ohlcv(symbol, timeframe, limit=limit)
        df = self._standardize(df)

        # Regime (Phase 2): computed on every decide for shadow logging +
        # meta-label features; only *gates the trigger set* when GATE_V2_ENABLED.
        regime_snap = classify_regime(df)
        vol_ok = self._vol_ok(df)

        # Compute raw indicators needed for Type2
        raw = self._compute_raw_indicators(df, timeframe)
        type2 = self._type2_rules(raw)

        # Collect Type1 direct signals from custom indicator plugins
        direct_signals = self._collect_direct_signals(df, raw=raw,
                                                      regime=regime_snap.regime)
        type1 = self._merge_direct_signals(direct_signals)

        # Combine heads using learned weights
        final_action, final_conf, blend_details = self._blend(type1, type2)

        regime_feats = regime_snap.feats()
        regime_feats["vol_ok"] = vol_ok
        out = IndicatorDecision(
            agent="indicator_agent",
            chartName=symbol, timeframe=timeframe,
            action=final_action,
            confidence=float(np.clip(final_conf, 0.0, 0.999)),
            details={
                "type1": type1,
                "type2": type2,
                "blend": blend_details,
                "direct_signals": direct_signals,
                "regime": regime_snap.regime,
                "regime_feats": regime_feats,
            }
        )
        if log:
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

    def apply_reward(self, blend: Dict[str, Any], reward: float):
        """Stateless RL update — replays the Type1/Type2 weight nudge from the
        PASSED ``blend`` snapshot (decide()'s details["blend"]) instead of
        re-reading the entire multi-MB predictions log on every call (the old
        learn() did, an O(file-size) operation). Also fixes the concurrency
        race: the graded prediction's own blend is applied, not whatever ran
        last.
        """
        pol = _load_policy()
        pol["score"] = pol.get("score", 0) + reward
        # Sign-anchored steps (v3.2.1): a CORRECT call is always +1, so its step
        # is the historical constant (0.03 type-share, +0.05 direct) — never
        # scaled down. A WRONG call anchors at -4 with the historical loss step
        # (0.03 type-share, -0.07 direct); the lesser losses (timeout -1.5,
        # missed -1) scale below it by |r|/4. This preserves the old dynamics
        # EXACTLY at the +1/-4 anchors while still letting the v2 map's graded
        # severities separate a wrong (-4) from a missed move (-1) — the prior
        # normalize-at-|r|=2 form made wins ~5x weaker than wrongs.
        r = float(reward)
        if r >= 0:
            ts_step, d_step = 0.03, 0.05
        else:
            mag = min(abs(r), 4.0)
            ts_step, d_step = -0.0075 * mag, -0.0175 * mag
        pol["weights"]["type1"] = float(np.clip(pol["weights"]["type1"] + ts_step*blend.get("type1_share", 0.5), 0.05, 0.95))
        pol["weights"]["type2"] = float(np.clip(pol["weights"]["type2"] + ts_step*blend.get("type2_share", 0.5), 0.05, 0.95))
        s = pol["weights"]["type1"] + pol["weights"]["type2"]
        pol["weights"]["type1"] = round(pol["weights"]["type1"]/s, 4)
        pol["weights"]["type2"] = round(pol["weights"]["type2"]/s, 4)
        fired = blend.get("fired_direct")
        if fired:
            d = pol["direct_signals"].get(fired, {"weight": 0.7, "score": 0})
            d["score"] += reward
            d["weight"] = float(np.clip(d["weight"] + d_step, 0.1, 0.95))
            pol["direct_signals"][fired] = d
        # v3.4 per-rule type-2 credibility: same sign-anchored step as the
        # direct signals, wider clip (base weight is 1.0, not 0.7). Only rules
        # that supported the graded action are touched; flag off => the blend
        # snapshot carries no fired_rules and this is a no-op.
        if config.T2_RULE_LEARNING:
            for key in blend.get("fired_rules") or []:
                d = pol.setdefault("type2_rules", {}).get(key, {"weight": 1.0, "score": 0})
                d["score"] += reward
                d["weight"] = float(np.clip(d["weight"] + d_step, 0.1, 2.0))
                pol["type2_rules"][key] = d
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

    def _compute_raw_indicators(self, df: pd.DataFrame, tf: Optional[str] = None) -> pd.DataFrame:
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
        
        # ✅ SuperTrend (length=10, multiplier=3 is common) — fast parity port
        # of pandas_ta.supertrend (the vendored per-row loop dominated decide())
        st = ci.supertrend_fast(out["high"], out["low"], out["close"], length=10, multiplier=3)
        out["supertrend"] = st["SUPERT_10_3.0"]
        out["supertrend_dir"] = st["SUPERTd_10_3.0"]   # 1 = bullish, -1 = bearish

        # RSI + OBV divergences (enhancement D1/D2): 4h/1d only — sub-4h
        # divergences are routinely faded; columns 0-filled (never NaN — the
        # type-2 dropna would eat the whole frame otherwise).
        if config.DIVERGENCE_VOTES and tf in ("4h", "1d"):
            out["rsi_div"] = float(ci.pivot_divergence(out["close"], out["rsi14"]))
            try:
                obv = ta.obv(out["close"], out["volume"])
                out["obv_div"] = float(ci.pivot_divergence(out["close"], obv))
            except Exception:
                out["obv_div"] = 0.0

        # v3.4 extra confluence votes — scalar broadcast columns like the
        # divergence pattern (0-filled, never NaN). Each key is opt-in via
        # T2_EXTRA_VOTES and stays off until it wins its backtest A/B.
        extra = config.T2_EXTRA_VOTES
        if extra:
            if "rsi30" in extra:
                out["v_rsi30"] = float(ci.rsi30_vote(close))
            if "mfi" in extra:
                out["v_mfi"] = float(ci.mfi_vote(out["high"], out["low"], close, out["volume"]))
            if "cci" in extra:
                out["v_cci"] = float(ci.cci_vote(out["high"], out["low"], close))
            if "vwap" in extra:
                out["v_vwap"] = float(ci.vwap_vote(out))
            if "ichimoku" in extra:
                out["v_ichimoku"] = float(ci.ichimoku_vote(out["high"], out["low"], close))
            if "fib" in extra:
                fib = ci.fib_confluence_vote(out)
                out["v_fib"] = float(fib["vote"])
                out["v_fib_ratio"] = float(fib["ratio"])

        return out

    def _type2_rules(self, raw: pd.DataFrame) -> Dict[str, Any]:
        r = raw.dropna().iloc[-1]

        # Every rule is named so per-rule credibility can be learned (v3.4):
        # (key, side, base) with side +1 bull / -1 bear, base = legacy vote size.
        rules: List[tuple] = []

        # MA ribbon
        if r["close"] > r["ma20"] and r["close"] > r["ma50"]:
            rules.append(("ribbon", 1, 2))
        elif r["close"] < r["ma20"] and r["close"] < r["ma50"]:
            rules.append(("ribbon", -1, 2))

        # RSI extremes
        if r["rsi14"] < 30:
            rules.append(("rsi14", 1, 1))
        elif r["rsi14"] > 70:
            rules.append(("rsi14", -1, 1))

        # MACD histogram
        rules.append(("macd", 1 if r["macd_hist"] > 0 else -1, 1))

        # BB squeeze-ish positioning bonus (lightweight)
        if r["close"] <= r["bb_lower"]:
            rules.append(("bb", 1, 1))
        elif r["close"] >= r["bb_upper"]:
            rules.append(("bb", -1, 1))

        # StochRSI (overbought/oversold)
        if r["stochrsi_k"] < 20 and r["stochrsi_d"] < 20:
            rules.append(("stochrsi", 1, 1))
        elif r["stochrsi_k"] > 80 and r["stochrsi_d"] > 80:
            rules.append(("stochrsi", -1, 1))

        # ✅ SuperTrend direction
        if r["supertrend_dir"] == 1:
            rules.append(("supertrend", 1, 2))
        elif r["supertrend_dir"] == -1:
            rules.append(("supertrend", -1, 2))

        # RSI / OBV divergence votes (D1/D2; columns exist only when
        # DIVERGENCE_VOTES is on and tf is 4h/1d)
        if "rsi_div" in r.index and r["rsi_div"]:
            rules.append(("rsi_div", 1 if r["rsi_div"] > 0 else -1, 1))
        if "obv_div" in r.index and r["obv_div"]:
            rules.append(("obv_div", 1 if r["obv_div"] > 0 else -1, 1))

        # v3.4 extra confluence votes (columns exist only for enabled
        # T2_EXTRA_VOTES keys); each ±1, recorded in `extras` for details.
        extras = {}
        for col in ("v_rsi30", "v_mfi", "v_cci", "v_vwap", "v_ichimoku", "v_fib"):
            if col in r.index:
                v = float(r[col])
                extras[col[2:]] = int(v)
                if v:
                    rules.append((col[2:], 1 if v > 0 else -1, 1))
        if "v_fib_ratio" in r.index and extras.get("fib"):
            extras["fib_ratio"] = float(r["v_fib_ratio"])

        # Tally. With T2_RULE_LEARNING each rule's base vote is scaled by its
        # learned credibility (default 1.0 => identical sums); with the flag
        # off the tally stays the legacy integer arithmetic, bit-identical.
        learn = config.T2_RULE_LEARNING
        t2w = self.policy.get("type2_rules", {}) if learn else {}
        votes = {"bull": 0, "bear": 0}
        for key, side, base in rules:
            w = base * float(t2w.get(key, {}).get("weight", 1.0)) if learn else base
            votes["bull" if side > 0 else "bear"] += w

        if votes["bull"] > votes["bear"]:
            action = "buy"
            confidence = 0.55 + 0.1*(votes["bull"] - votes["bear"])
        elif votes["bear"] > votes["bull"]:
            action = "sell"
            confidence = 0.55 + 0.1*(votes["bear"] - votes["bull"])
        else:
            action = "skip"
            confidence = 0.45

        out = {
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
        if extras:
            out["extras"] = extras       # shape unchanged when no extra votes on
        if learn and action in ("buy", "sell"):
            # rules that supported the winning side — the ones apply_reward
            # credits/debits (mirrors fired_direct semantics for type-1)
            want = 1 if action == "buy" else -1
            out["fired_rules"] = [k for k, side, _ in rules if side == want]
        return out

    def _vol_ok(self, df: pd.DataFrame) -> bool:
        """Last closed volume above its SMA — the gate's volume confirmation."""
        try:
            vol = df["volume"].astype(float)
            sma = vol.rolling(config.VOLUME_SMA_LEN).mean().iloc[-1]
            if pd.isna(sma):
                return True  # not enough history to judge -> don't block
            return bool(vol.iloc[-1] > float(sma))
        except Exception:
            return True

    def _collect_direct_signals(self, df: pd.DataFrame, raw: Optional[pd.DataFrame] = None,
                                regime: Optional[str] = None) -> List[Dict[str, Any]]:
        """Type-1 direct signals. With GATE_V2_ENABLED and a trending regime the
        mean-reversion NWE is excluded and the trend triggers (supertrend flip,
        Donchian breakout, squeeze release) take its place; otherwise the legacy
        set runs unchanged (flag off => byte-identical to pre-Phase-2)."""
        signals = []
        trending = (config.GATE_V2_ENABLED
                    and regime in ("trend_up", "trend_down"))

        if not trending:
            # Ensure indicator columns exist
            df = ci.apply_nadaraya_watson_envelope(df, h=self.nwe_h, mult=self.nwe_mult)
            # NWE_EVENT_MODE (correctness v3, A3): crossing-triggered signal —
            # fires once when price crosses out of the band instead of every
            # bar it stays outside (state mode re-fires for hours).
            nwe_fn = (ci.direct_signal_from_nwee if config.NWE_EVENT_MODE
                      else ci.direct_signal_from_nwe)
            sig = nwe_fn(df)
            if sig:
                sig["name"] = "nwe"
                signals.append(sig)

        # --- Chandelier Exit ---
        df = ci.chandelier_exit(df)   # adds long_stop, short_stop, ce_signal columns
        latest = df.iloc[-1]
        if latest['ce_signal'] in ["buy", "sell"]:
            signals.append({
                "signal": latest['ce_signal'],
                "confidence": _direct_conf("chandelier_exit", 0.9),
                "name": "chandelier_exit"
            })

        df = ci.alpha_trend(df)
        latest = df.iloc[-1]
        if latest['alpha_signal'] in ["buy", "sell"]:
            signals.append({
                "signal": latest['alpha_signal'],
                "confidence": _direct_conf("alpha_trend", 0.9),
                "name": "alpha_trend"
            })

        if trending:
            # Trend-regime triggers (Phase 2)
            st = supertrend_flip_from_raw(raw) if raw is not None else None
            if st:
                signals.append(st)
            dc = ci.donchian_breakout_signal(df)
            if dc and dc["signal"] in ("buy", "sell"):
                signals.append(dc)
            sq = ci.squeeze_release_signal(df)
            if sq and sq["signal"] in ("buy", "sell"):
                signals.append(sq)

        # Each entry: {"signal":"buy"/"sell"/"skip", "confidence": float, "name": ...}
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
        if type2.get("fired_rules") is not None:
            # snapshot travels prediction -> grader -> apply_reward, stateless
            blend_details["fired_rules"] = type2["fired_rules"]
        return action, confidence, blend_details
