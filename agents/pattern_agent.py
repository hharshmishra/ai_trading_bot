import random

def get_pattern_signal():
    # Dummy chart pattern recognition
    pattern = random.choice(["Head & Shoulders", "Doji", "Engulfing", "None"])
    bias = random.choice(["Bullish", "Bearish", "Neutral"])
    confidence = round(random.uniform(0.5, 1.0), 2)
    return {"agent": "pattern", "pattern": pattern, "bias": bias, "confidence": confidence}
