# v3.7 — Learning Repair + Gate Upgrade (19-day production audit)

**Evidence window:** Oracle production Jul 5 → Jul 24 2026, 27,312 predictions,
26,976 graded, 292 emitted to Telegram. Grading integrity was verified against
live Binance klines on a 20-signal stratified sample: **20/20 exact matches**
(labels, triple-barrier hits, returns to 5 decimals). The plumbing was healthy —
zero downtime gaps >2.5h, grading latency median 38s, zero overdue, zero
duplicates. The learning was not.

## What went wrong

Emitted directional accuracy decayed **55% → 39% → 16% → 14%** week over week.

| Defect | Mechanism | Prod evidence |
|---|---|---|
| Shift-normalized trust | `weight = (score − min) / Σ` measures *distance from the worst agent* | indicator (lead voter) → weight **2.3e-06** within 24h; the **disabled** sentiment voter (score frozen at +1.5 because Δ = reward × conf and its conf was always 0) held **top trust 0.355** |
| Asymmetric trust reward | brain Δ used the bandit map (−4 wrong / +1 correct / −1.5 flat) × conf | at realized base rates (~29% up / 29% down / 41% flat) every directional voter bankrupts: scores −286 … −1525, unbounded, unrecoverable (~30k net-correct to climb back) |
| Unbounded bandit weights | REINFORCE step with no clamp (news lr 0.1) | news logits ±287, research ±68, derivatives ±53 — softmax saturated, policies deterministic beyond ε |
| Grader fetch window | `_path_after` fetched a fixed 50 recent candles | an outage >47h silently grades 1h rows against the wrong window |
| Nightly trainer | in-process 02:00 IST loop, no marker | a stop spanning 02:00 silently skips that night's training |
| Confidence saturation | conf = \|score\|/Σw·conf → 1.0 exactly on unanimity | emitted conf==1.0 rows graded **1 correct / 7 wrong / 15 flat** — herding is an anti-signal |

The golden pattern — indicator BUY + research BUY with news/derivatives
abstaining (`conf_over_80` buys) — hit **10/11 on direction** early in the run,
then died with indicator's trust.

## What v3.7 changes

- **Trust:** `weights = softmax(score / 2)`, scores clamped **[−10, +10]**
  (self-heals legacy files on load), exact **2% per-agent floor**; `decide()`
  renormalizes over the *active* roster so flag-disabled agents hold no vote
  mass. Trust reward is now symmetric direction-quality (+1·conf / −1·conf /
  −0.25·conf flat / 0 for skip votes), decoupled from the bandit map.
- **Bandits:** per-weight clamp ±5 in all four RL updates; news lr 0.1 → 0.05.
- **Grader:** fetch sized from elapsed time (50-bucketed, cap 1000); a fully
  post-entry window stays pending instead of mislabeling.
- **Nightly:** `logs/nightly_marker.json` last-success marker (always written);
  `NIGHTLY_CATCHUP=true` trains once at startup when the latest 02:00 IST
  boundary went unserved.
- **Gate:** `META_GATE_ENABLED=true` goes live (emitted counterfactual:
  meta_p ≥ 0.55 → **37.5%** hit vs **17.8%** below; deciles near-monotone;
  holdout AUC 0.579 is under the documented 0.60 bar — enabled on the
  emitted-subset lift, env-revertible, retrains nightly).
  `GATE_CONF_SATURATION=0.97` suppresses unanimity herds.
  `GATE_1H_MIXED=false` retires the 2/17 `nwe_mixed` path (flag restores it).
- **Reset:** `scripts/reset_learning.py --policies-only` — archives + deletes
  the six policy files and four nightly artifacts, **keeps the DB learning
  tables** (meta training data) and line logs. Full reset chosen deliberately:
  the math fix alone, applied to the old score vector, would still hand the
  sentiment agent ~92% trust — the reset is load-bearing, not hygiene.
- **Repo:** learned policy JSONs untracked (`logs/*.json` ignored top-level;
  backtest reports stay tracked).

Post-reset trust weights: `{indicator .343, research .208, news .126,
derivatives .162, sentiment .162}`.

## Deploy notes (manual, on Oracle)

Order matters — **archive before any git operation**:

1. `sudo systemctl stop bitreinforcex`
2. Archive the learned state (only copy on earth):
   `D=logs/archive/$(date -u +%F); mkdir -p $D; cp logs/*policy.json logs/calibration.json logs/meta_model.pkl logs/meta_metrics.json logs/indicator_conf.json logs/bitreinforcex.db $D/`
3. `git checkout -- logs/ && git pull --ff-only` (the untracking commit removes
   the six policy JSONs from the tree; your archive keeps them)
4. `venv/bin/python scripts/reset_learning.py --policies-only` (review plan),
   then re-run with `--yes`. **Never run the full mode on Oracle** — it wipes
   predictions/outcomes/rewards and starves the meta model below
   `META_MIN_ROWS` for weeks.
5. Append to `.env` (14 parity keys that were silently missing — the live file
   predates commit `754451d` which added them to `.env.example` — plus the
   v3.7 flags):

   ```
   # --- v3.7 learning repair + gate upgrade ---
   SENTIMENT_ENABLED=true
   SENTIMENT_TTL_SECONDS=3300
   SENTIMENT_BRAIN_SCORE=1.5
   T2_EXTRA_VOTES=
   T2_RULE_LEARNING=false
   REGIME_ADX_LEN=14
   REGIME_CHOP_LEN=14
   REGIME_VOL_LOOKBACK=100
   REGIME_WALK_BARS=50
   REWARD_CORRECT=1.0
   REWARD_WRONG=-4.0
   DERIV_OI_WINDOW_H=6
   CALIBRATION_MIN_ROWS_PER_TF=150
   PRED_RETENTION_D=120
   META_GATE_ENABLED=true
   META_GATE_THRESHOLD=0.55
   GATE_CONF_SATURATION=0.97
   GATE_1H_MIXED=false
   NIGHTLY_CATCHUP=true
   ```
6. `venv/bin/python scripts/preflight.py` → READY, then
   `sudo systemctl start bitreinforcex`. **Check the `env parity` line** — it
   names every `.env.example` key missing from the box's environment. If step 5
   was pasted incompletely it says so here instead of costing another 19 days.
7. Expect within minutes: a `nightly catch-up:` line (marker absent →
   immediate training regenerates `meta_model.pkl`/`calibration.json`, so the
   meta gate is never silently inert), a fresh `logs/brain_policy.json` at the
   default weights above, and a sentiment vote in the first `:30` cycle.
8. **Security:** if the leaked Bot D token was never rotated — BotFather →
   `/revoke` → update `.env` → restart.

## 48h monitoring

- sentiment: `SELECT COUNT(*) FROM predictions WHERE sentiment_action IS NOT NULL AND candle_close_ts > strftime('%s','now')-86400;` > 0; `SELECT COUNT(*) FROM rewards WHERE agent='sentiment';` climbing after horizons pass
- trust: `logs/brain_policy.json` scores within ±10, weights in [0.02, 0.92]
- bandits: max |weight| ≤ 5 in all four policy files
- meta gate: emitted directional rows have `meta_p >= 0.55`
- saturation: zero emitted rows with `final_confidence >= 0.97`
- mixed: zero emitted rows with `trigger_source='nwe_mixed'`
- volume: ~8–10 emitted/day (from ~15) — the meta gate passed 72/162 of the
  old emitted set
- `logs/nightly_marker.json` refreshes after every 02:00 IST

Suppressed rows record `emitted=0` with `trigger_source=NULL` (the suppress
reason is not persisted) — monitor via the `meta_p` / `final_confidence`
columns, not `trigger_source`.

## Why it stayed invisible for 19 days

Nothing checked that a deployed `.env` still matched `.env.example`. The
sentiment flag entered the example on Jul 5 (`754451d`), hours before the
production box was deployed from an `.env` that predated it; `config.py`
supplied `False` and the voter cast zero votes for 19 days without a single
warning. v3.6's commit titled "env parity" only made the two `.env.example`
files identical — it added no runtime guard.

v3.7 adds `_env_parity()` to `scripts/preflight.py`: it reads the key names
from `.env.example`, diffs them against the live environment, and **warns**
(never fails — a box may omit optional keys deliberately) naming each missing
key. Presence is the test, not truthiness, so `T2_EXTRA_VOTES=` counts as set.
Values are never read or printed. Run preflight after any `.env` edit.

## Known follow-ups

- Scores may pin at the ±10 rails within days of heavy feedback; the 2% floor
  keeps minority voices alive. If pinning proves sticky, add score decay.
- README recommends `T2_RULE_LEARNING=on` at a go-live reset; v3.7 pins it
  `false` per the operator's choice — revisit after the repair stabilizes.
- Historical rows mis-graded by the old fixed-50 fetch remain in `outcomes`
  (small volume; identifiable by grade-time gaps) — accepted.
