# BitReinforceX

> *"Reinforcing your trades with AI power"*

An AI crypto **signal** system (advisory only — it never places orders). Four
reinforcement-learning agents feed a decision-making brain; a regime classifier
gates which triggers may fire; signals go to Telegram; and the system **trains
itself** by grading its own past predictions against realized price with
triple-barrier labels — then trains a nightly meta-model on its own track
record.

## Architecture

```
 RSS (8 feeds) ─▶ ingestion (hourly) ─▶ RAG (rag.py: embed + dedup + retrieve)
                                        │ headlines
 scheduler (IST :30) ─▶ market context (once/cycle: LLM + F&G + BTC dominance)
                                        │
   per pair ─▶ brain ─▶ News · Indicator(+Regime) · Research · Derivatives
                                        │ decision (4-voter weighted sum)
              regime-gated signal gate (signals.py v2) ─▶ Telegram
                                        │ record (persistence.py: SQLite)
   grader (60s) ─▶ triple-barrier labels ─▶ rewards ─▶ all 4 agent policies
   nightly (02:00 IST) ─▶ meta-label model + confidence calibration
```

| Module | Role |
|--------|------|
| `agents/` | News (OpenAI + RAG), Indicator (NWE/Chandelier/AlphaTrend + type-2 + trend triggers), Research (5 macro logics), **Derivatives** (funding/OI/long-short positioning, no LLM), **Regime** (ADX/CHOP classifier — gates, doesn't vote) |
| `brain/decision_maker.py` | Weighted 4-voter aggregation + learned agent priorities |
| `signals.py` | Gate v1 + regime-gated gate v2 (truth table), cascade, formatting |
| `grading/barriers.py` | Triple-barrier labeler (shared by grader AND backtest) |
| `grader.py` | Auto-labels predictions; TB reward map v2; manual override wins |
| `backtest/` | Replays the production decide()+gate per-bar over history |
| `jobs/` | Nightly meta-label training + per-TF isotonic confidence calibration |
| `market_context.py` / `cycle.py` / `persistence.py` / `telegram_app.py` / `rag.py` / `ingestion.py` | Shared context, cycle orchestration, SQLite state, Telegram runtime, headline RAG |

## Quickstart

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt            # installs the vendored pandas_ta too
cp .env.example .env                        # then fill in keys/tokens
python scripts/preflight.py                 # verify everything is wired
python telegram_app.py                      # run (signals bot + scheduler + grader + nightly)
```

Tests: `pytest -q` (107 tests, no network/keys needed — mocks + HashingEmbedder).

## Accuracy upgrade (v2) — evidence-first rollout

The 1h baseline backtest measured **33% TB precision** for counter-trend NWE
signals ("band walk"). v2 fixes this with a regime classifier and ships every
change **behind a flag, off by default**:

| Flag | What it enables | Enable when |
|------|-----------------|-------------|
| `GATE_V2_ENABLED` | Regime-gated gate: NWE only in ranging regimes; Supertrend-flip / Donchian / squeeze triggers own trending regimes; volume confirmation | backtest A/B beats baseline per-TF with CI separation |
| `TB_GRADING_ENABLED` | Triple-barrier rewards (ATR-scaled TP/SL/time; labels always recorded) | after TB label distribution sanity-checks vs backtest |
| `DERIVATIVES_ENABLED` | 4th voter from Binance USDM positioning ($0, keyless) | immediately safe; bandit learns from graded rows |
| `META_GATE_ENABLED` | Gate on nightly meta-model p(correct) | holdout AUC ≥ 0.60 AND +5pts precision over 4-week shadow |

Backtest (exact mode — replays the real `IndicatorAgent.decide` + gate):

```bash
python scripts/run_backtest.py --pairs BTCUSDT,ETHUSDT,SOLUSDT --tfs 1h,4h --start 2024-07-01
python scripts/run_backtest.py --pairs all --tfs 1h --start 2024-07-01 --gate v2 \
       --baseline logs/backtest/baseline/report.json --workers 6
python scripts/run_training.py             # manual nightly-training run
```

Caveat printed in every report: news/research aren't backtestable (no
historical LLM) — the confidence-gate path uses indicator-only confidence as a
proxy; NWE/trend trigger paths are exact.

## How the learning loop works

Every prediction is stored with each agent's RL replay payload, barrier prices
(`entry ± mult × ATR`), and a grade-due time. The grader walks the realized
OHLC path: first barrier touched decides the label (TP / SL / timeout) — a
TP-then-crash sequence grades by what happened first. Rewards (map v2):
correct +1, wrong direction −4, directional-but-flat −1.5, missed move −1.0.
Manual feedback from the dev channel still overrides (corrections net out).

Nightly at 02:00 IST the system trains on its own graded history: a
meta-label model p(correct | regime, agreement, positioning, confidence…) and
per-TF isotonic calibration (runtime applies JSON knots via `np.interp` — no
sklearn on the hot path). Both run in shadow (stamped on rows, shown in dev
dumps) until their enable criteria are met.

## Signals

Customer channel gets the signal; dev channel gets signal + brain dump +
feedback buttons + regime/trigger/derivatives context lines. Gate v1 (default):
1h emits only on a direct NWE signal; other TFs emit on confidence ≥ 80% OR
NWE (NWE wins conflicts). Gate v2 (flag): the truth table in
`signals.should_emit_signal_v2` — NWE owns ranging, trend triggers own
trending, 1h stays NWE-only-in-ranging (confidence alone never emits on 1h).

## Deploy (Oracle Cloud ARM / any VPS)

```bash
sudo cp deploy/bitreinforcex.service /etc/systemd/system/   # edit User=/paths
sudo systemctl daemon-reload && sudo systemctl enable --now bitreinforcex
journalctl -u bitreinforcex -f
```

State is one SQLite file (WAL). Back it up nightly with `deploy/backup.sh` (cron)
or continuously with Litestream (`deploy/litestream.yml`).

## Notes

- **pandas_ta is vendored** (`vendor/`) because upstream vanished from PyPI and
  GitHub. It pins `numpy<2` / `pandas<3`.
- **sklearn is pinned** (`scikit-learn==1.3.2`, `scipy==1.11.4`): newer versions
  drag in numpy 2.x and break the pandas 2.0.3 ABI. `preflight.py` asserts this.
- **CryptoPanic free API tier ended April 2026** — ingestion is RSS-only (8 feeds).
- **RAG embeddings**: default `HashingEmbedder` needs no extra deps; MiniLM
  option documented in `requirements.txt`.
- **Cost**: still ~85 LLM calls/cycle at 48 pairs on `gpt-4o-mini` — the regime,
  derivatives, meta and calibration layers add **zero** LLM calls.
