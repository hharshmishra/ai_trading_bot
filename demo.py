from typing import Any, Dict, Tuple

js = {'action': 'sell', 'confidence': 0.6523061327927014, 'raw': {'agent': 'indicator_agent', 'chartName': 'ETHUSDT', 'timeframe': '4h', 'action': 'sell', 'confidence': 0.6523061327927014, 'details': {'type1': {'action': 'sell', 'confidence': 0.99, 'scores': {'buy': 0.0, 'sell': 0.41364851044848977, 'skip': 0.0}, 'used': [{'name': 'nwe', 'signal': 'sell', 'confidence': 0.5666417951349175, 'weight': 0.73, 'contribution': 0.41364851044848977}], 'fired_direct': 'nwe'}, 'type2': {'action': 'buy', 'confidence': 0.98, 'votes': {'bull': 5, 'bear': 0}, 'last_row': {'close': 4748.05, 'ma20': 4464.528, 'ma50': 4391.629, 'rsi14': 66.29383422190415, 'macd_hist': 41.837207255202756, 'bb_lower': 4007.7128164257183, 'bb_upper': 4921.343183574282, 'stochrsi_k': 71.74623185769245, 'stochrsi_d': 70.40789836682002, 'supertrend': 4451.12798948784, 'supertrend_dir': 1}}, 'blend': {'type1_weight': 0.65, 'type2_weight': 0.35, 'type1_share': 0.65, 'type2_share': 0.35, 'scores': {'buy': 0.34299999999999997, 'sell': 0.6435, 'skip': 0.0}, 'fired_direct': 'alpha_trend'}, 'direct_signals': [{'signal': 'sell', 'confidence': 0.5666417951349175, 'name': 'alpha'}, {'signal': 'buy', 'confidence': 0.7, 'name': 'nwe'}]}}}

def extract_nwe_signal(agent_result: Dict[str, Any]) -> Tuple[str | None, float | None]:
    """Return (signal, confidence) from IndicatorAgent's NWE direct signal if present."""
    try:
        details = agent_result.get("raw").get("details") or {}
        ds = details.get("direct_signals")
        # print(ds)
        if not ds:
            return None, None
        # direct_signals may be a list of {name, signal, confidence} or a dict keyed by name
        if isinstance(ds, dict):
            nde = ds.get("nwe") or ds.get("NWE")
            if isinstance(nde, dict):
                sig = (nde.get("signal") or nde.get("action") or "").lower() or None
                conf = nde.get("confidence")
                return sig, float(conf) if conf is not None else None
        elif isinstance(ds, list):
            for item in ds:
                if not isinstance(item, dict):
                    continue
                nm = str(item.get("name", "")).lower()
                if nm == "nwe":
                    sig = (item.get("signal") or item.get("action") or "").lower() or None
                    conf = item.get("confidence")
                    return sig, float(conf) if conf is not None else None
    except Exception:
        pass
    return None, None

if __name__ == "__main__":
    sig, sconf = extract_nwe_signal(js)
    print(sig, " is sig, and, ", sconf, " is sconf")