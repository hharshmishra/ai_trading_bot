<div align="center">

# ⚡ BitReinforceX

**Self-training crypto signal system — five RL agents, one brain, evidence-gated everything.**

*"Reinforcing your trades with AI power"*

![Python](https://img.shields.io/badge/python-3.11-blue?logo=python&logoColor=white)
![Tests](https://img.shields.io/badge/tests-436%20passing-brightgreen)
![LLM](https://img.shields.io/badge/LLM-gpt--4o--mini-8A2BE2)
![Cost](https://img.shields.io/badge/data%20cost-%240%2Fmo-success)
![Deploy](https://img.shields.io/badge/deploy-Oracle%20ARM%20%2B%20systemd-orange)
![License](https://img.shields.io/badge/signals-advisory%20only-red)

[Architecture](#-architecture) · [Quickstart](#-quickstart) · [How it learns](#-how-it-learns) ·
[Signal gate](#-signal-gate) · [Backtesting](#-backtesting) · [Configuration](#%EF%B8%8F-configuration) ·
[Deploy](#-deploy) · [Docs](#-deeper-docs)

</div>

---

> **Advisory only.** BitReinforceX emits trading *signals* to Telegram. It never places orders.

## 🧠 What it is

Five reinforcement-learning agents analyze 48 Binance USDT pairs every hour and vote; a
decision brain aggregates them with learned priorities; a **regime-gated signal gate** decides
what's worth sending; a grader later checks every prediction against realized price with
**triple-barrier labels** and rewards each agent for what *it* said. Nightly, the system trains a
meta-model and confidence calibration **on its own track record**.

| Agent | Signal source | LLM? | Learns via |
|---|---|---|---|
| 📰 **News** | RAG-grounded headlines (8 RSS feeds), typed events | gpt-4o-mini | 10-feature softmax bandit |
| 📈 **Indicator** | NWE · Chandelier · AlphaTrend · trend triggers · 8 type-2 rules | no | learned per-indicator weights + type1/type2 blend |
| 🔬 **Research** | ecosystem membership · SPX+DXY price/news · money-flow · BTC dominance | gpt-4o-mini | 10-feature softmax bandit |
| 🧲 **Derivatives** | funding rate · open interest · top-trader long/short (Binance USDM, keyless) | no | 8-feature softmax bandit |
| 🌡 **Sentiment** | Fear&Greed level/trend/extremes · mempool fee pressure · tx momentum · price-vs-usage divergence · CoinGecko trending · taker buy/sell flow (raw klines) | no | 10-feature softmax bandit |
| 📐 **Regime** *(gate, not voter)* | ADX + Choppiness + vol-percentile, hysteresis | no | deterministic |

## 🏗 Architecture

```
 RSS (8 feeds) ─▶ ingestion (hourly) ─▶ RAG (embed · dedup · retrieve)
                                            │ [age][tier] headlines
 scheduler (IST :30 == UTC :00) ─▶ market context (once/cycle: overall news,
        │ UTC-aligned cascade        SPX+DXY trends, dominance ROC, F&G, drivers)
        ▼                                   │
   per pair ─▶ BRAIN ─▶ News · Indicator(+Regime) · Research · Derivatives · Sentiment
                 │            weighted vote  Σ wᵢ·aᵢ·confᵢ
                 ▼
     regime-gated signal gate ─▶ Telegram (customer + dev+buttons channels)
                 │ every prediction recorded (SQLite WAL)
                 ▼
   grader (60s): triple-barrier first-touch labels ─▶ rewards ─▶ 5 policies
   nightly (02:00 IST): meta-label model · confidence calibration ·
                        empirical-Bayes indicator confidences · ecosystem refresh
```

**Key invariant:** the backtest harness replays the *production* `decide()` + gate functions
bar-by-bar — live and simulation share one code path, closed candles only, UTC-aligned.

## 🚀 Quickstart

```bash
git clone https://github.com/hharshmishra/ai_trading_bot.git && cd ai_trading_bot
python3.11 -m venv venv && source venv/bin/activate
pip install -r requirements.txt        # installs the vendored pandas_ta too
cp .env.example .env                   # fill in: OPENAI_API_KEY, TELEGRAM_BOT_TOKEN, chat IDs
python scripts/preflight.py            # must print READY (pins, DB, universe freshness, .env parity)
python telegram_app.py                 # runs scheduler + grader + nightly in one process
```

Tests (no network / keys needed): `pytest -q` → **436 passing**.

## 🎓 How it learns

1. **Every prediction is recorded** with each agent's replayable feature snapshot, entry price
   (true candle close), and ATR-scaled barrier prices (`TP = entry ± 1.5·ATR`, `SL = ∓1.0·ATR`).
2. **The grader walks the realized path**: first barrier touched decides — TP-then-crash counts
   by what happened *first*. Rewards: correct **+1**, wrong direction **−4**, directional-but-flat
   **−1.5**, missed move **−1.0**.
3. **Manual override wins — one tap**: dev-channel verdict buttons (`BUY / SELL / FLAT`)
   re-grade any signal in a single press — every agent AND the brain are trained against your
   verdict with the active reward map (FLAT included: skip-callers rewarded, directional calls
   get the timeout penalty). A claim CAS prevents the auto-grader and a human racing on the
   same row; corrections net the policy to the human verdict.
4. **Nightly self-training**: logistic meta-model p(correct | regime, agreement, positioning, …),
   per-TF isotonic confidence calibration (runtime = `np.interp` on JSON knots — zero sklearn on
   the hot path), and shrunk win-rate confidences replacing hardcoded ones. All shadow-first.
5. **Brain trust (v3.8)**: agent priority = `softmax(score/2)` with scores clamped ±10 and a 2%
   floor — trust learns an *advantage* signal, `conf·(outcome − per-agent EMA baseline)`, plus a
   nightly decay toward 0, so a bad market regime can't sink every voter to the rails (the v3.7
   symmetric map was still negative-sum at real base rates: 21 days pinned derivatives at −10).
   Post-mortems: [docs/v37-learning-repair.md](docs/v37-learning-repair.md),
   [docs/v38-emission-redesign.md](docs/v38-emission-redesign.md).
6. **Edge-first emission (v3.8)**: every candidate signal (`nwe`, `sms*`, `trend`, `conf`) is
   judged by the **evidence ledger** — measured hit-rate ≥ 38% AND Wilson lower bound ≥ 30% in its
   `(source, tf, regime, vol)` cohort — then by the meta gate. Suppressed candidates are still
   recorded + graded, so cohorts earn their way in (or out) without emitting. Hand-tuned per-source
   flags stopped deciding anything; 21 days of them had suppressed the one proven signal
   (NWE crossings, 40–50% outside calm vol) to ~1.5 emissions/day.

## 🚦 Signal gate

Regime decides which trigger family may emit (v2, backtest-validated):

| TF | Ranging / Mixed | Trending |
|---|---|---|
| **1h** | NWE **crossing** + volume (+ brain agreement in mixed). Confidence alone **never** emits. | NWE suppressed. Trend triggers off by default. |
| **4h / 1d / 1w** | conf ≥ 0.80 (NWE needs `GATE_NWE_HIGHER_TF` — measured 12.5%, off) | aligned **trend_continuation** + volume, or trend-aligned conf ≥ 0.80. Counter-trend suppressed. |

**Signal times (IST)** — real UTC candle closes: 1h every hour at :30 · 4h at 1:30/5:30/9:30/13:30/17:30/21:30 · 1d at 5:30 · 1w Monday 5:30.

Evidence (12 pairs × 2y exact replay, reports in `logs/backtest/`): trending-regime NWE
(~20k emissions at 23–31% precision) eliminated; `trend_continuation` 39.7–41.9% at 4h with
positive expectancy; 4h trending conf **+7.6/+4.4pts (significant)**; NWE event mode **−25–36%**
duplicate 1h emissions at +0.8–1.1pp precision.

## 🔬 Backtesting

```bash
# replay the production pipeline over history (CSV-cached, closed candles)
python scripts/run_backtest.py --pairs BTCUSDT,ETHUSDT --tfs 1h,4h --start 2024-07-01
# A/B any flag against an archived baseline
NWE_EVENT_MODE=true python scripts/run_backtest.py --gate v2 --pairs all --tfs 1h \
    --start 2024-07-01 --label my-experiment --baseline logs/backtest/candidate-v3/report.json
python scripts/analyze_indicators.py   # redundancy matrix + standalone win rates
python scripts/run_training.py         # manual nightly-training run
```

Every report prints its honest caveat: news/research aren't backtestable (no historical LLM) —
the confidence path uses indicator-only confidence as proxy; NWE/trend paths are exact.

## ⚙️ Configuration

**Adding a coin is one line** — `UNIVERSE_ADD=APTUSDT` in `.env` (or one ticker in
[`universe.py`](universe.py)). News tagging derives automatically; RL policies are global
(per-agent, not per-pair) so the new pair is scored, graded and learned from starting with its
first cycle; `preflight.py` fails loudly on dead/typo'd symbols. Optional polish: an
`ingestion.ALIASES` entry and ecosystem membership.

All knobs live in [`.env.example`](.env.example) (40+ keys, every one commented) and parse in
[`config.py`](config.py). Rollout philosophy: **ship dark → measure → enable on evidence**.

<details>
<summary><b>Current flag states (what to enable, what's collecting evidence)</b></summary>

| Flag | State | Why |
|---|---|---|
| `GATE_V2_ENABLED` `TB_GRADING_ENABLED` `DERIVATIVES_ENABLED` | ✅ on | backtest-validated (v2) |
| `CLOSED_CANDLES_ONLY` `NEWS_RAG_ENABLED` | ✅ on | correctness fixes (v3) |
| `NWE_EVENT_MODE` | ✅ on | −25–36% duplicate emissions, precision non-inferior |
| `MACRO_PRICES_ENABLED` | ✅ on | SPX via stooq keyless; DXY once `FRED_API_KEY` set |
| `BRAIN_DEADZONE_V2` | 🌒 shadow | enable if suppressed cohort shows negative expectancy (≥2wk) |
| `EMISSION_V2_ENABLED` | ✅ on (v3.8) | edge-first gate: evidence ledger (rate≥0.38 ∧ WilsonLB≥0.30, n≥25 per cohort) + meta ranker. Seeded from 21d history; off restores the v3.7.1 truth table byte-for-byte ([evidence](docs/v38-emission-redesign.md)) |
| `SMS_ENABLED` / `SMS_EMIT` | ✅ on / 🌒 shadow (v3.8) | Smart Money Structure port (BOS/CHoCH/momentum label + trend matrix). Backtest (39k events, pre-registered rule): base+0.2–1.5pts → records + grades but does NOT emit; day-20 prod data re-judges |
| `META_GATE_ENABLED` | ✅ on (v3.7) | meta_p≥0.55 as ranker on candidates. v3.8 fixed the train/serve skew (`emitted` leakage feature dropped, candidate one-hots persisted on every row) that had it serving 0.97s on a 37% cohort |
| `GATE_CONF_SATURATION` | ✅ 0.97 (v3.7) | conf==1.0 unanimity herds graded 1c/7w/15f emitted — suppressed (0 = off) |
| `GATE_1H_MIXED` | ⚰️ superseded (v3.8) | the n=17 verdict was refuted at n=96 (40.6% hit) — the ledger owns regime/vol fit now; flag only matters with `EMISSION_V2_ENABLED=false` |
| `NIGHTLY_CATCHUP` | ✅ on (v3.7) | runs a missed 02:00 IST training once at startup |
| `GATE_NWE_VOL_MAX` | ⚰️ superseded (v3.8) | 21d data inverted the premise: NWE is BEST in elevated/extreme vol (45.7/42.6%) and worst in calm (23.5%) — exactly what the ledger's vol-band cohorts encode |
| `MONEY_FLOW_V2` `NEWS_EVENTS_ENABLED` `ECOSYSTEMS_AUTO` | ⏸ off | enable after shadow sanity |
| `DIVERGENCE_VOTES` `GATE_TREND_REVERSAL` `GATE_NWE_HIGHER_TF` `GATE_1H_TREND` | ❌ off | measured: no benefit / harmful |
| `EMPIRICAL_DIRECT_CONF` | ⏸ off | self-arms once ≥30 graded direct-fires exist |
| `T2_EXTRA_VOTES` (rsi30/mfi/cci/vwap/fib/ichimoku) | ❌ off | measured 2026-07-05: no significant edge, all six refused promotion ([evidence](docs/v34-vote-evidence.md)) |
| `T2_RULE_LEARNING` | ⏸ off | v3.4 per-rule type-2 credibility — recommended ON at go-live reset; learns each rule's weight from real outcomes |
| `SENTIMENT_ENABLED` | ✅ on (v3.7) | v3.5 5th voter (F&G/on-chain/taker flow, all $0) — was silently absent from the prod `.env` for 19 days (0 votes); enabled with the v3.7 reset. Feature IC study in [docs/sentiment-evidence.md](docs/sentiment-evidence.md) |

</details>

## 📦 Deploy

```bash
# Oracle Cloud ARM (or any VPS) — one process, one systemd unit, one SQLite file
sudo cp deploy/bitreinforcex.service /etc/systemd/system/   # edit User= and paths
sudo cp deploy/logrotate.conf /etc/logrotate.d/bitreinforcex # runtime.log rotation
sudo systemctl daemon-reload && sudo systemctl enable --now bitreinforcex
journalctl -u bitreinforcex -f
```

Backups: nightly `deploy/backup.sh` (WAL-safe) or continuous Litestream (`deploy/litestream.yml`).

**Reset learning to zero** (before a real go-live, after test runs polluted the record): `python scripts/reset_learning.py` (dry-run) then `--yes`. Backs up everything it wipes; keeps market-data cache and the subscriptions DB. **v3.7 repair reset:** `--policies-only` wipes only policies + nightly artifacts and KEEPS the DB learning tables (the meta model's training set) — this is the mode to use on a live deployment.

> ⚠️ **Pinned stack — do not casually `pip install`:** `numpy==1.25.0`, `pandas==2.0.3`
> (vendored `pandas_ta` in `vendor/` — upstream is deleted from PyPI/GitHub),
> `scikit-learn==1.3.2`, `scipy==1.11.4` (newer pulls numpy 2.x and breaks the ABI).
> `preflight.py` asserts all of it, plus that every universe pair still trades
> (it has already caught LUNA, LRC and MKR delistings) and that every
> `.env.example` key is present in the environment — a flag missing from a
> deployed `.env` silently falls back to its code default, which is how the
> 5th voter sat dead in production for 19 days.

## 💼 Renting it out (membership mode)

Optional subscription layer (default **off**): 7/15/30-day paid plans for the signals
channel, an independent Pro plan for direct agent access via the control bot, INR (UPI)
+ USDT (TRC-20) rails with **no webhook server**, and fully automated Telegram access —
join-request gating on payment, auto-kick after expiry + grace. Design, pricing and
economics: [`docs/subscription-deck.html`](docs/subscription-deck.html). Enable with the
`MEMBERSHIP_*` keys in [`.env.example`](.env.example).

## 📚 Deeper docs

| Doc | What's inside |
|---|---|
| [`docs/overview-deck.html`](docs/overview-deck.html) | **Start here (non-technical)** — plain-English business overview: the 5 AI analysts, how a signal works, what a subscriber gets, and how the subscription earns |
| [`docs/dev-deck.html`](docs/dev-deck.html) | **The developer deck** — full system walkthrough 0→100: every module, every env var, every flow, runbook |
| [`docs/accuracy-upgrade.html`](docs/accuracy-upgrade.html) | v2 evidence deck — baseline vs gate-v2 numbers, truth table, rollout criteria |
| [`docs/system-design.html`](docs/system-design.html) | v1 rebuild deck — original architecture story |
| [`docs/telegram-deck.html`](docs/telegram-deck.html) | Telegram operations deck — bots, commands, admin flows, message formats |
| [`docs/subscription-deck.html`](docs/subscription-deck.html) | Subscription business deck — plans, pricing, payment rails, lifecycle |
| `logs/backtest/*/report.md` | Every archived backtest run (baseline, ablation, candidates) |
| [`tasks/todo.md`](tasks/todo.md) / [`tasks/lessons.md`](tasks/lessons.md) | Project log + hard-won lessons |

---

<div align="center">
<sub>Telegram: customer channel (signals) · dev channel (signals + brain dump + award/punish buttons) ·
control bot (<code>/news /indicator /research /context /regime /derivs</code>)</sub>
</div>
