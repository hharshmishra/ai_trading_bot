# AI Trading Bot - Phase 1

## Overview
This is the Phase-1 implementation of an AI-based trading bot with multiple agents:
- **News Analysis Agent**
- **Technical Indicator Agent**
- **Chart Pattern Agent**
- **Decision Maker**

Currently, the agents return dummy/mock data to simulate functionality. In later stages, you can replace them with actual implementations.

## Folder Structure
```
ai_trading_bot/
├── agents/
│   ├── news_agent.py
│   ├── indicator_agent.py
│   └── pattern_agent.py
├── brain/
│   └── decision_maker.py
├── logs/
│   └── predictions_log.json
├── main.py
├── requirements.txt
└── README.md
```

## How to Run
1. Create a virtual environment and activate it:
```bash
python3 -m venv venv
source venv/bin/activate
```

2. Install dependencies:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

3. Run the bot:
```bash
python main.py
```

## Output
The bot will print and log a JSON-formatted trading signal.

Example:
```json
{ "chartName": "BTCUSDT", "signal": "BUY", "entry": 118000, "accuracy": 0.91 }
```
