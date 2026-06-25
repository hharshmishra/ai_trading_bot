"""Shared market-context builder (Phase 1).

The single biggest cost defect in the old system: every one of the ~48 pairs
re-ran the *same* market-wide work. Per pair, `research_agent.decide` called the
news agent on the ecosystem drivers + SPX + DXY and the indicator agent on the
money-flow basket + BTC/BTCDOM — and the brain re-ran the news overall scan.
The market-wide pieces (overall sentiment, SPX, DXY, money-flow phase, BTC
dominance, per-ecosystem driver trends) are IDENTICAL for every coin in a cycle,
yet were recomputed 48×. That was ~576 LLM calls/cycle, mostly redundant.

This module computes all of that ONCE per (cycle, timeframe) into a
``MarketContext``. ``research_agent.decide(market_context=ctx)`` then reads from
it instead of fanning out per coin, collapsing the cost to ~73 calls/cycle.

Equivalence by construction
---------------------------
We do NOT re-implement the market logic here. We call Research's own pure logic
methods (`_logic2_spx`, `_logic3_money_flow`, `_logic4_btcdominance`,
`_logic5_dxy`) and reuse its scoring helpers, so the shared values are bit-for-bit
what the per-coin path would have produced. A thin `_SharedOverallNews` wrapper
injects the one shared overall scan into every news call so SPX/DXY/driver
sentiment each cost 1 LLM call (pair scan) instead of 2.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agents.research_agent import (
    ECOSYSTEMS,
    ECOSYSTEM_DRIVERS,
    _strip_suffix,
    _sent_to_score,
    _action_to_score,
)


def _ind_action_conf(out: Any) -> tuple[Optional[str], float]:
    """Normalise an IndicatorAgent output (dataclass OR dict) to (action, conf)."""
    if hasattr(out, "action"):
        return getattr(out, "action", None), float(getattr(out, "confidence", 0.6) or 0.6)
    if isinstance(out, dict):
        return out.get("action"), float(out.get("confidence", 0.6) or 0.6)
    return None, 0.6


class _SharedOverallNews:
    """Wraps a NewsAgent so every ``run(pair)`` reuses one shared overall scan.

    Passed to Research's news-driven logics (SPX/DXY) and to the driver loop, so
    the market-wide overall scan is computed once, not once per news call.
    """

    def __init__(self, news_agent: Any, overall_json: Dict[str, Any]):
        self._news = news_agent
        self._overall = overall_json

    def run(self, pair: str) -> Dict[str, Any]:
        return self._news.run(pair, overall_json=self._overall)


def _drivers_for_symbols(symbols: List[str]) -> List[str]:
    """The minimal set of ecosystem-driver base tickers needed by these symbols.

    Mirrors ResearchAgent._logic1_ecosystem driver selection exactly: primary
    ecosystem = first match; drivers = ECOSYSTEM_DRIVERS[eco] or ECOSYSTEMS[eco][:3];
    the coin itself is excluded as a driver of its own ecosystem.
    """
    needed: set[str] = set()
    for sym in symbols:
        base = _strip_suffix(sym)
        ecos = [eco for eco, members in ECOSYSTEMS.items()
                if base in [m.upper() for m in members]]
        if not ecos:
            continue
        eco = ecos[0]
        drivers = ECOSYSTEM_DRIVERS.get(eco, ECOSYSTEMS.get(eco, [])[:3])
        for d in drivers:
            if d.upper() != base:
                needed.add(d.upper())
    return sorted(needed)


@dataclass
class MarketContext:
    """All market-wide signals for one (cycle, timeframe), computed once."""
    timeframe: str
    overall_json: Dict[str, Any]
    spx_score: float
    dxy_score: float
    money_flow_phase: float
    btdom_effect: float
    driver_ind_score: Dict[str, float] = field(default_factory=dict)   # base -> [-1,1]
    driver_news_score: Dict[str, float] = field(default_factory=dict)  # base -> [-1,1]
    details: Dict[str, Any] = field(default_factory=dict)

    def eco_scores(self, base: str, eco: str) -> tuple[List[float], List[float]]:
        """Return (indicator_scores, news_scores) for an ecosystem's drivers,
        excluding ``base`` — the exact inputs Logic 1 averages."""
        drivers = ECOSYSTEM_DRIVERS.get(eco, ECOSYSTEMS.get(eco, [])[:3])
        drivers = [d.upper() for d in drivers if d.upper() != base][:3]
        ind = [self.driver_ind_score[d] for d in drivers if d in self.driver_ind_score]
        news = [self.driver_news_score[d] for d in drivers if d in self.driver_news_score]
        return ind, news


def build_market_context(
    timeframe: str,
    symbols: List[str],
    indicator_agent: Any,
    news_agent: Any,
    research_agent: Any,
) -> MarketContext:
    """Compute the shared market context once for ``timeframe`` and ``symbols``.

    LLM cost: 1 (overall) + 1 (SPX) + 1 (DXY) + 1 per distinct ecosystem driver.
    Indicator/OHLCV work is de-duplicated by the per-cycle cache in data_fetcher.
    ``research_agent`` is used only to invoke its pure market-logic helpers, so
    the resulting values match the per-coin path exactly.
    """
    # 1) One shared market-wide overall scan (1 LLM call).
    overall_json = news_agent.scan_overall().model_dump()
    shared_news = _SharedOverallNews(news_agent, overall_json)

    # 2) Global news-driven signals (1 LLM each), via Research's own logic.
    spx_score, spx_details = research_agent._logic2_spx(shared_news)
    dxy_score, dxy_details = research_agent._logic5_dxy(shared_news)

    # 3) Global indicator-driven signals (no LLM; OHLCV cached).
    money_flow, mf_details = research_agent._logic3_money_flow(timeframe, indicator_agent, None)
    btdom, btd_details = research_agent._logic4_btcdominance(timeframe, indicator_agent, None)

    # 4) Per-distinct-driver indicator trend + news sentiment (1 LLM per driver).
    driver_ind_score: Dict[str, float] = {}
    driver_news_score: Dict[str, float] = {}
    driver_raw: Dict[str, Any] = {}
    for base in _drivers_for_symbols(symbols):
        dp = base + "USDT"
        try:
            out = indicator_agent.decide(dp, timeframe)
            act, conf = _ind_action_conf(out)
            driver_ind_score[base] = _action_to_score(act, conf)
            driver_raw.setdefault(base, {})["indicator"] = {"action": act, "confidence": conf}
        except Exception as e:  # a single bad driver must not sink the cycle
            driver_raw.setdefault(base, {})["indicator_error"] = str(e)
        try:
            r = shared_news.run(dp)
            pj = r.get("pair_json", {})
            driver_news_score[base] = _sent_to_score(pj.get("sentiment"), pj.get("confidence"))
            driver_raw.setdefault(base, {})["news"] = {
                "sentiment": pj.get("sentiment"), "confidence": pj.get("confidence")}
        except Exception as e:
            driver_raw.setdefault(base, {})["news_error"] = str(e)

    return MarketContext(
        timeframe=timeframe,
        overall_json=overall_json,
        spx_score=float(spx_score),
        dxy_score=float(dxy_score),
        money_flow_phase=float(money_flow),
        btdom_effect=float(btdom),
        driver_ind_score=driver_ind_score,
        driver_news_score=driver_news_score,
        details={
            "spx": spx_details, "dxy": dxy_details,
            "money_flow": mf_details, "btcdominance": btd_details,
            "drivers": driver_raw,
        },
    )
