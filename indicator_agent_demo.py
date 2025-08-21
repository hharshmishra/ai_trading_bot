# demo_indicator.py
from agents.indicator_agent import IndicatorAgent
import json
from dataclasses import asdict, is_dataclass


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


if __name__ == "__main__":
    # CSV first. If you want live data, set prefer_csv=False (requires ccxt).
    agent = IndicatorAgent(prefer_csv=False)  # change to False for live ccxt

    symbol = "LINKUSDT"
    timeframe = "1h"          # ccxt e.g. "1m","5m","15m","1h","4h","1d"
    raw = agent.decide(symbol, timeframe)
    cleaned = structure_output(raw)
    print(cleaned)
    # name = input("Please enter true outcome: ")

    # --- later, after you know the outcome, call learn()
    # Suppose our action made profit -> reward:
    # agent.learn(predicted_action=raw.action, true_outcome=name)   # or "sell"/"skip"
