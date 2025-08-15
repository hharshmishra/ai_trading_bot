import json
from datetime import datetime

def decide(news_data, indicator_data, pattern_data):
    # Dummy decision logic: if 2/3 suggest bullish -> BUY else SELL
    bullish_count = 0
    if news_data.get("sentiment") == "Bullish":
        bullish_count += 1
    if indicator_data.get("signal") == "BUY":
        bullish_count += 1
    if pattern_data.get("bias") == "Bullish":
        bullish_count += 1

    final_signal = "BUY" if bullish_count >= 2 else "SELL"

    result = {
        "chartName": "BTCUSDT",
        "signal": final_signal,
        "entry": 118000,  # Dummy entry price
        "accuracy": 0.91, # Dummy accuracy
        "timestamp": datetime.utcnow().isoformat()
    }

    # Log result to file
    with open("logs/predictions_log.json", "a") as log_file:
        log_file.write(json.dumps(result) + "\n")

    return result
