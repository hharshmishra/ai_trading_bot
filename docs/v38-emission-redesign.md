# v3.8 — Edge-First Emission + Smart Money Structure + Learning Repair II

**Evidence window:** Oracle production Jul 24 → Aug 14 2026 (21 days,
31,728 graded predictions, grading re-verified 20/20 against live Binance
klines via ccxt — labels, triple-barrier hits, and returns exact).

## What the 21-day audit found

Plumbing: perfect. Zero downtime gaps >2.5h, grading latency avg 38s / max
84s / zero overdue, nightly trainer ran every day, sentiment voted and was
rewarded, macro snapshots fresh. OpenAI cost $3.89 for the whole window.

Everything else:

| Finding | Evidence |
|---|---|
| Emission collapse | 31 emissions in 20 days (~1.5/day), zero on 6 of the last 11 days |
| Funnel | ~800 candidates → meta_gate killed 446, low_volume 197, mixed/higher-tf flags 111 → 31 sent |
| Emitted quality | ~29% hit ≈ the 28.5% random directional base; 4h trend_continuation sells 1/15 |
| All five voters ≤ random | hit%: indicator 25.4, research 27.2, news 28.5, deriv 25.9, sentiment 24.1 — and wrong% > hit% by 5–8pts for every one |
| Brain trust rail-pinned | symmetric ±conf reward is negative-sum at real base rates (E ≈ −0.11·conf/vote): all scores sank, derivatives pinned at −10, research held 64% mass by being least-bad |
| **The edge** | event-mode NWE crossings: **40.6–50% hit outside calm vol** (n=145; calm 23.5 / normal 47.8 / elevated 45.7 / extreme 42.6 vs ~30.6% base). When the brain disagreed with a crossing, the crossing won 46.5% vs 25.6% |
| Own flags refuted | `GATE_1H_MIXED=false` blocked 96 crossings @ 40.6% (decided on n=17); `GATE_NWE_HIGHER_TF=false` blocked 16 @ 43.8% |
| Meta train/serve skew | cycle passed mid-gate `emitted`/`trigger_source` at serve time; training rows carried them only when finally emitted → meta_p 0.97–1.0 streaks on a ~37% cohort, 0.095 on candidates. `emitted` was leakage outright |
| Sent-vs-graded split | emitted rows were TB-graded on the brain's `final_action`, not the direction Telegram carried (e.g. ALGOUSDT sent SELL, graded as BUY/tp); 9 emitted rows had no recorded direction at all |

## What v3.8 changes

1. **Meta feature-set v2** — `emitted` dropped (leakage); trigger one-hots read
   the new `candidate_trigger` column, persisted on **every** candidate row
   pre-suppression, so train == serve by construction. New features:
   `candidate_side` (emitted buys 7/14 vs sells 5/32), `vol_pct × nwe`,
   SMS strength/confidence/CVD.
2. **Grade what was sent** — emitted rows anchor TB barriers and labels on
   `candidate_action` (the direction the subscriber received). Non-emitted
   shadow rows keep the old path; old rows are untouched (NULL fallback).
3. **Trust advantage baseline** — `Δ = conf · (outcome − per-agent EMA)`
   (`TRUST_BASELINE_ALPHA=0.02`) ends the negative-sum sink; nightly
   `TRUST_DECAY=0.98` pulls scores toward 0 so a rail is never permanent.
4. **Edge-first emission (`EMISSION_V2_ENABLED`)** — candidates (`nwe`,
   `sms`, `sms_bos`, `sms_choch`, `trend`, `conf`) are judged by the
   **evidence ledger** (`logs/emission_ledger.json`, rebuilt nightly): per
   `(source, tf, regime_group, vol_band)` cohort, emit only when measured
   hit-rate ≥ `LEDGER_FLOOR` (0.38) **and** Wilson LB ≥ `LEDGER_LB_GUARD`
   (0.30) at n ≥ 25. Unmeasured cohorts fall back to the source-global test;
   brand-new sources get a bounded probation (`LEDGER_PROBATION_N=40`).
   Suppressed candidates are still recorded and graded, so cohorts grow and
   self-heal **without emitting**. The meta gate (`meta_p ≥ 0.55`) still
   applies on top. The old regime truth-table remains fully intact behind
   `EMISSION_V2_ENABLED=false`.
   - Seeded from the 21-day history: `nwe|1h` global (42.6% @ n=129) and
     `nwe|1h|mixed|elevated` (45.2% @ 62) emit; `nwe|1h|mixed|normal`
     (40.7% @ 27, LB 24.5%) waits for n; `conf` (8.1%) and `trend` (5.9%)
     are dead until they earn otherwise. Expected volume ~2.5–4/day.
5. **Smart Money Structure** (port of "Smart Money Structure | GainzAlgo") —
   three sources: `sms` (vol-adaptive momentum + volume expansion + breakout),
   `sms_bos` / `sms_choch` (colored-candle structure crossings on strict
   pivots), plus the trend matrix (EMA20 + daily VWAP on 1h/4h/1d, strength
   −100..+100, confidence tier, normalized CVD) shown on every signal and fed
   to the meta model. **Backtest verdict (pre-registered rule): SMS_EMIT=false**
   — 39,368 events over 12 pairs / 2 years landed base +0.2–1.5pts, so SMS
   runs in **shadow** (recorded, graded, ledgered — never sent) until the
   day-20 audit re-judges it on realized production data.
6. **Signal message** now carries: model p(correct), the emitting source's
   track record (n, hit-rate, LB), the SMS trend matrix, regime.
7. Hygiene: news actions persisted lowercase; a reset now also clears the
   nightly marker so `NIGHTLY_CATCHUP` retrains the (reshaped) meta model at
   the very next startup instead of serving a stale artifact until 02:00.

## Deploy (manual, on Oracle)

Box reality (verified Aug 14): repo at `/home/ubuntu/Desktop/ai_trading_bot`,
bot running as a bare `python telegram_app.py` inside the orphaned RDP desktop
session, **no systemd unit**. This deploy also moves it under systemd so a
desktop session is never load-bearing again. Order matters — archive first.

```bash
cd ~/Desktop/ai_trading_bot

# 1. archive the learned state (the only copy on earth)
D=logs/archive/$(date -u +%F); mkdir -p $D
cp logs/*policy.json logs/calibration.json logs/meta_model.pkl \
   logs/meta_metrics.json logs/indicator_conf.json logs/nightly_marker.json \
   logs/bitreinforcex.db $D/

# 2. stop the bot (it lives in the desktop session, not systemd)
pkill -f telegram_app.py
pgrep -af telegram_app.py     # must print nothing

# 3. pull v3.8
git checkout -- logs/ 2>/dev/null; git pull --ff-only

# 4. policies-only reset (keeps the DB: it is the meta model's and the
#    ledger's training data). NEVER the full mode on this box.
venv/bin/python scripts/reset_learning.py --policies-only        # review
venv/bin/python scripts/reset_learning.py --policies-only --yes

# 5. seed the evidence ledger from the box's own 21-day history
venv/bin/python scripts/seed_ledger.py            # review the cohort table
venv/bin/python scripts/seed_ledger.py --yes
```

6. Append to `.env` (comments on their own lines only — systemd's
   EnvironmentFile does not strip inline comments):

   ```
   # --- v3.8 edge-first emission + SMS ---
   EMISSION_V2_ENABLED=true
   LEDGER_FLOOR=0.38
   LEDGER_LB_GUARD=0.30
   LEDGER_MIN_N=25
   LEDGER_PROBATION_N=40
   TRUST_DECAY=0.98
   SMS_ENABLED=true
   SMS_EMIT=false
   SMS_PIVOT_LEN=5
   SMS_MOMENTUM_BASE=0.01
   SMS_MIN_DIST=5
   ```

7. `venv/bin/python scripts/preflight.py` → **READY**; check the
   `env parity` line (every key present) and the `emission ledger` line
   (cohort count, not a WARN).

8. Install the systemd unit with corrected paths, and drop `EnvironmentFile`
   (the app loads `.env` itself via python-dotenv — that removes the
   inline-comment hazard class entirely):

   ```bash
   sed -e 's#/home/ubuntu/ai_trading_bot#/home/ubuntu/Desktop/ai_trading_bot#g' \
       -e '/^EnvironmentFile=/d' deploy/bitreinforcex.service \
       | sudo tee /etc/systemd/system/bitreinforcex.service
   sudo systemctl daemon-reload
   sudo systemctl enable --now bitreinforcex
   systemctl is-active bitreinforcex    # active
   ```

9. Verify within minutes, in the journal (`journalctl -u bitreinforcex -f`):
   - `nightly catch-up:` line (marker was reset → immediate training builds
     the v2-feature meta model + calibration + ledger refresh);
   - a fresh `logs/brain_policy.json` with a `baseline` block;
   - the first emissions carry `TRACK RECORD` / `MARKET STRUCTURE` lines.
   Then kill the PID once (`sudo systemctl kill bitreinforcex` or `kill <pid>`)
   and watch systemd restart it.

10. Optional hardening, now safe (bot no longer lives in the desktop):
    xrdp `KillDisconnected=true` + `DisconnectedTimeLimit=3600` in
    `/etc/xrdp/sesman.ini`, and a `Restart=on-failure` drop-in for xrdp.

11. **Security (still outstanding):** if the leaked Bot D token was never
    rotated — BotFather → `/revoke` → update `.env` → restart.

## 48h monitoring

- volume: ~2–4 emissions/day; every emitted row has `gate_reason` = its
  source (`nwe`, …) and `meta_p >= 0.55`;
- `SELECT gate_reason, COUNT(*) FROM predictions WHERE created_ts > strftime('%s','now')-86400 GROUP BY 1;`
  → suppressions now say `ledger_below_floor` / `ledger_cold` / `meta_gate` /
  `sms_shadow`;
- `SELECT COUNT(*) FROM predictions WHERE candidate_trigger LIKE 'sms%';`
  climbing (shadow recording works);
- `logs/emission_ledger.json` refreshes nightly; `logs/brain_policy.json`
  scores hover near 0 instead of sinking (baseline block present);
- no `ledger_missing` reasons after the seed.

## Day-20 protocol (next audit, ~Sep 3)

1. **Funnel:** `SELECT gate_reason, emitted, COUNT(*) FROM predictions GROUP BY 1,2`
   — plus outcomes joined per cohort (`candidate_trigger`, `candidate_action`
   make every suppressed candidate gradeable now).
2. **Ledger verdicts realized:** for each cohort, compare the ledger's
   seeded LB to the realized 20-day hit-rate — the gate's calibration,
   measured directly.
3. **SMS re-judgment:** hit-rate of `candidate_trigger LIKE 'sms%'` rows by
   cohort. Flip `SMS_EMIT=true` only if a cohort passes the same two-part
   test (rate ≥ 0.38, LB ≥ 0.30, n ≥ 25) on production data.
4. **Trust health:** brain scores bounded away from the rails, baselines
   near each agent's real outcome average, weights not monopolized.
5. **Meta gate:** with skew fixed, meta_p deciles on candidates should be
   monotone; if AUC on the candidate subset stays < 0.55, consider dropping
   the meta gate to ranker-only (log, don't veto).
6. Decide: widen the ledger (arm `sms`?), tune floors, or hold.

## Rollback

`EMISSION_V2_ENABLED=false` in `.env` + restart returns the complete v3.7.1
emission behavior (regime truth table + all v3.7 flags). The new columns and
features are additive and harmless either way.
