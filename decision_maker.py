from agents.research_agent import ResearchAgent
from agents.indicator_agent import IndicatorAgent
from agents.news_agent import NewsAgent

import json

def restructure_json(raw_data: dict) -> dict:
    structured = {
        "agent": raw_data.get("agent"),
        "chart": {
            "name": raw_data.get("chartName"),
            "timeframe": raw_data.get("timeframe"),
            "action": raw_data.get("action"),
            "confidence": raw_data.get("confidence")
        },
        "details": {
            "features": {
                "eco": {
                    "score": raw_data["details"]["features"].get("eco_score"),
                    "news": raw_data["details"]["features"].get("eco_news"),
                    "ind": raw_data["details"]["features"].get("eco_ind"),
                },
                "spx_news": raw_data["details"]["features"].get("spx_news"),
                "money_flow_phase": raw_data["details"]["features"].get("money_flow_phase"),
                "btc_dominance_effect": raw_data["details"]["features"].get("btcdominance_effect"),
                "dxy_news": raw_data["details"]["features"].get("dxy_news"),
                "trend": {
                    "child": raw_data["details"]["features"].get("child_trend"),
                    "parent": raw_data["details"]["features"].get("parent_trend"),
                    "bias": raw_data["details"]["features"].get("bias"),
                }
            },
            "logic": {
                "logic1": {
                    "ecosystems": raw_data["details"]["logic_details"]["logic1"].get("ecosystems", []),
                    "primary": raw_data["details"]["logic_details"]["logic1"].get("primary"),
                    "drivers": raw_data["details"]["logic_details"]["logic1"].get("drivers", []),
                    "indicator_votes": raw_data["details"]["logic_details"]["logic1"].get("indicator_driver_votes", []),
                    "news_votes": raw_data["details"]["logic_details"]["logic1"].get("news_driver_votes", []),
                    "external_ecosystem_sent": raw_data["details"]["logic_details"]["logic1"].get("external_ecosystem_sent"),
                },
                "logic2": raw_data["details"]["logic_details"].get("logic2"),
                "logic3": raw_data["details"]["logic_details"].get("logic3"),
                "logic4": raw_data["details"]["logic_details"].get("logic4"),
                "logic5": raw_data["details"]["logic_details"].get("logic5"),
            }
        }
    }
    return structured






research = ResearchAgent(prefer_csv=False)

sym = "ETCUSDT"
tf = "4h"
res_out = research.decide(sym, tf, indicator_agent=IndicatorAgent(), news_agent=NewsAgent())

print(json.dumps(restructure_json(res_out), indent=2))