# BitReinforceX

> *"Reinforcing your trades with AI power"*

An AI crypto **signal** system (advisory only — it never places orders). Three
reinforcement-learning agents feed a decision-making brain; signals go to
Telegram; and the system **trains itself** by grading its own past predictions
against realized price.

## Architecture

```
 RSS / CryptoPanic ─▶ ingestion ─▶ RAG (rag.py: embed + dedup + retrieve)
                                        │ headlines
 scheduler (IST :30) ─▶ market context (build once/cycle, Phase 1 cost fix)
                                        │
       per pair ─▶ brain (decision_maker) ─▶ News · Indicator · Research agents
                                        │ decision
                          signal gate (signals.py) ─▶ Telegram (telegram_app.py)
                                        │ record (persistence.py: SQLite)
                          grader (grader.py, every 60s) ─▶ realized price ─▶ reward
```

| Module | Role |
|--------|------|
| `agents/` | News (OpenAI + RAG), Indicator (NWE/Chandelier/AlphaTrend + 6 type-2), Research (5 macro logics) — each a contextual-bandit RL learner |
| `brain/decision_maker.py` | Weighted aggregation + learned agent priorities |
| `market_context.py` | Builds market-wide signals **once per cycle** (≈6× fewer LLM calls) |
| `cycle.py` | One analysis cycle: context → 48 pairs → gate → broadcast → record |
| `persistence.py` | One SQLite DB: predictions, outcomes, rewards, sessions, news corpus |
| `grader.py` | Auto-labels predictions from realized price; manual Telegram feedback overrides |
| `signals.py` | Signal gate + scheduler cascade + message formatting |
| `telegram_app.py` | Long-lived bot(s): signals + dev feedback buttons + control commands |
| `rag.py` / `ingestion.py` | Headline embedding/dedup/retrieval to ground the LLM |

## Quickstart

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt            # installs the vendored pandas_ta too
cp .env.example .env                        # then fill in keys/tokens
python scripts/preflight.py                 # verify everything is wired
python telegram_app.py                      # run (signals bot + scheduler + grader)
```

Tests: `pytest -q` (22 tests, no network/keys needed — uses mocks + the
HashingEmbedder).

## How the learning loop works

Every prediction is stored with each agent's RL replay payload and a
**grade-due time** (`candle_close + k` candles). The grader fetches the realized
candle, computes the forward return vs a per-timeframe threshold, and rewards
each agent (+1 correct / −4 wrong) against *its own* recorded prediction — no
human needed. In the dev Telegram channel you can override any signal; manual
feedback takes precedence (a correction nets out a prior auto-grade).

Grading defaults (tunable in `grader.py`): horizon `k` = 3/2/1/1 and threshold
= 0.4%/1.0%/2.5%/5% for 1h/4h/1d/1w.

## Signals

Sent to a customer channel (signal only) and a dev channel (signal + brain dump
+ feedback buttons). Gating: 1h emits only on a direct NWE signal; other
timeframes emit on confidence ≥ 80% OR a direct NWE signal (NWE wins on
conflict).

## Deploy (Oracle Cloud ARM / any VPS)

```bash
sudo cp deploy/bitreinforcex.service /etc/systemd/system/   # edit User=/paths
sudo systemctl daemon-reload && sudo systemctl enable --now bitreinforcex
journalctl -u bitreinforcex -f
```

State is one SQLite file (WAL). Back it up nightly with `deploy/backup.sh` (cron)
or continuously with Litestream (`deploy/litestream.yml`).

## Notes

- **pandas_ta is vendored** (`vendor/`) because upstream was removed from PyPI
  and GitHub. It pins `numpy<2` / `pandas<3`.
- **RAG embeddings**: the default `HashingEmbedder` needs no extra deps. For real
  semantic similarity (MiniLM), see the optional section in `requirements.txt`.
- **Cost**: ~85 LLM calls/cycle at 48 pairs (down from ~488) on `gpt-4o-mini`.
