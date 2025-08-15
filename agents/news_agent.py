import random

def get_news_signal():
    # Dummy news sentiment score
    sentiment = random.choice(["Bullish", "Bearish", "Neutral"])
    score = round(random.uniform(0.5, 1.0), 2)
    return {"agent": "news", "sentiment": sentiment, "score": score}
