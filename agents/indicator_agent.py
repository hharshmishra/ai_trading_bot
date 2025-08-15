import random

def get_indicator_signal():
    # Dummy indicator signal
    signal = random.choice(["BUY", "SELL", "HOLD"])
    strength = round(random.uniform(0.5, 1.0), 2)
    return {"agent": "indicator", "signal": signal, "strength": strength}
